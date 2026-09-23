"""The orchestrator loop over a real queue and store, with the fake benchmark and example policies."""

from __future__ import annotations

import json
import threading

import pytest

from conftest import FAKE_PIN, fake_spec_doc
from duel_helpers import (
    REPLAY,
    REPLAY_REF,
    ZERO,
    ZERO_REF,
    Crash,
    FakePolicyRuntime,
    RecordingReporter,
    harness_voiding,
)
from store_helpers import make_record, publish
from vector_orchestrator.benchmarks.units import plugin_units
from vector_orchestrator.canon import Signer
from vector_orchestrator.daemon import Daemon
from vector_orchestrator.duel.orchestrate import DuelRequest, Orchestrator
from vector_orchestrator.ids import SubmissionRef
from vector_orchestrator.queue import Queues
from vector_orchestrator.spec import load_schema
from vector_orchestrator.store.verify import Report, SchemaCheck, verify_store
from vector_orchestrator.store.writer import Store, store_lock

TRACK = "franka_1arm"


@pytest.fixture
def signer():
    return Signer.generate()


@pytest.fixture
def store(duel_spec, tmp_path, signer, fake_installed):
    store = Store(tmp_path / "store", duel_spec, signer)
    store.init(signer.verify_key_hex)
    return store


@pytest.fixture
def queues(duel_spec, tmp_path):
    return Queues(tmp_path / "queue", duel_spec.tracks)


def daemon(spec, store, tmp_path, runtime=None, **kwargs):
    """A daemon as a fresh process would make one: its own store object, runtime and reporter."""
    fresh = Store(store.root, spec, store.signer)
    orchestrator = Orchestrator(
        spec,
        fresh,
        runtime or FakePolicyRuntime(spec),
        tmp_path / "runs",
        live=RecordingReporter(spec),
        **kwargs,
    )
    return Daemon(orchestrator, Queues(tmp_path / "queue", spec.tracks), idle_sleep_s=0)


def add(queues, ref, size="smoke"):
    queues[TRACK].add(ref.repo, ref.revision, duel_size=size)


def test_an_empty_throne_is_taken_by_genesis_then_the_next_entry_duels(
    duel_spec, store, queues, tmp_path
):
    add(queues, ZERO_REF)
    add(queues, REPLAY_REF)
    loop = daemon(duel_spec, store, tmp_path)
    assert loop.step(TRACK) is True
    head = store.head(TRACK)
    assert head["king"] == ZERO_REF.as_dict() and head["block"] == 1
    assert loop.step(TRACK) is True
    head = store.head(TRACK)
    assert head["king"] == REPLAY_REF.as_dict() and head["block"] == 2
    kinds = [r["kind"] for r in store.iter_index(TRACK)]
    assert kinds == ["genesis", "duel"]
    assert loop.step(TRACK) is False, "an empty queue ran something"
    state = queues[TRACK].reload()
    assert state.entries == [] and state.in_progress is None and state.block == 2

    snapshot = json.loads(store.queue_path(TRACK).read_text())
    report = Report()
    SchemaCheck(load_schema()).check("QueueSnapshot", snapshot, "queue.json", report)
    assert report.errors == []
    assert verify_store(store.root, duel_spec).ok


def test_a_killed_daemon_restarted_finishes_the_duel_running_each_unit_once_per_side(
    duel_spec, store, queues, tmp_path
):
    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, REPLAY_REF)
    # Killed while the king's side runs: the challenger's three units and one of the king's done.
    killed = daemon(duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, crash_on_serve=4))
    with pytest.raises(Crash):
        killed.step(TRACK)
    state = queues[TRACK].reload()
    assert state.entries == [] and state.in_progress is not None, "the duel was lost"
    assert len(store.iter_index(TRACK)) == 1

    runtime = FakePolicyRuntime(duel_spec)
    restarted = daemon(duel_spec, store, tmp_path, runtime)
    restarted.run(once=True)
    assert queues[TRACK].reload().in_progress is None
    records = store.iter_index(TRACK)
    assert [r["kind"] for r in records] == ["genesis", "duel"] and records[1]["dethroned"]
    run_dir = tmp_path / "runs" / TRACK / records[1]["event_id"][:16]
    units = store.event(TRACK, records[1]["event_id"])["units"]
    for side in ("challenger", "king"):
        for unit in units:
            runs = (run_dir / side / unit["unit_id"] / "runs.log").read_text()
            assert runs.count("\n") == 1, (
                f"{side} {unit['unit_id']} ran {runs.count(chr(10))} times"
            )
    assert len(runtime.serves) == 2, "the restarted daemon re-ran finished units"
    assert verify_store(store.root, duel_spec).ok


def test_a_resumed_duel_whose_king_lost_the_crown_requeues_its_challenger_and_never_publishes(
    duel_spec, store, queues, tmp_path
):
    """The daemon is killed mid-duel; while it is down, a duel on the command line dethrones the
    king. The resumed duel must not publish against the old king, which would crown him again."""
    other = SubmissionRef.make("org/other-policy", "6" * 40)
    local = {REPLAY_REF.repo: REPLAY, ZERO_REF.repo: ZERO, other.repo: REPLAY}
    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, REPLAY_REF, size="smoke")
    killed = daemon(
        duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, local, crash_on_serve=4)
    )
    with pytest.raises(Crash):
        killed.step(TRACK)
    stale = queues[TRACK].reload().in_progress.event_id

    by_hand = daemon(duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, local)).orchestrator
    assert by_hand.run(DuelRequest(TRACK, other, ZERO_REF, "smoke", block=3)).record["dethroned"]

    restarted = daemon(duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, local))
    assert restarted.step(TRACK) is True
    state = queues[TRACK].reload()
    assert state.in_progress is None
    assert [(e.key, e.duel_size) for e in state.entries] == [(REPLAY_REF.key, "smoke")]
    assert (tmp_path / "runs" / TRACK / f"{stale[:16]}.stale-1").is_dir()
    assert [r["kind"] for r in store.iter_index(TRACK)] == ["genesis", "duel"]

    assert restarted.step(TRACK) is True
    records = store.iter_index(TRACK)
    assert [(r["king"]["key"], r["challenger"]["key"]) for r in records[1:]] == [
        (ZERO_REF.key, other.key),
        (other.key, REPLAY_REF.key),
    ]
    assert records[2]["block"] == 4 and store.head(TRACK)["king"] == other.as_dict()
    assert verify_store(store.root, duel_spec).ok


def test_a_duel_resumed_while_its_benchmark_is_missing_stays_in_progress_for_a_later_try(
    spec_doc, write_spec, duel_spec, store, queues, tmp_path
):
    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, REPLAY_REF)
    killed = daemon(duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, crash_on_serve=4))
    with pytest.raises(Crash):
        killed.step(TRACK)
    taken = queues[TRACK].reload().in_progress

    doc = fake_spec_doc(spec_doc, pin={**FAKE_PIN, "version": "9.9.9"})  # mid-upgrade
    doc["budgets"]["act_timeout_s"] = 2.0
    upgrading = write_spec(doc, name="upgrading.json")
    assert daemon(upgrading, store, tmp_path).step(TRACK) is False
    assert queues[TRACK].reload().in_progress == taken, "the resumed duel was dropped"

    runtime = FakePolicyRuntime(duel_spec)
    assert daemon(duel_spec, store, tmp_path, runtime).step(TRACK) is True
    assert [r["kind"] for r in store.iter_index(TRACK)] == ["genesis", "duel"]
    assert len(runtime.serves) == 2, "the finished units were lost"
    assert queues[TRACK].reload().in_progress is None


def test_a_duel_the_hub_or_docker_cannot_serve_stays_in_progress_for_a_later_try(
    duel_spec, store, queues, tmp_path
):
    from vector_orchestrator.duel.runtime import RuntimeUnavailable

    class Down(FakePolicyRuntime):
        def fetch(self, ref, *, workdir):
            raise RuntimeUnavailable("docker is not running")

    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, REPLAY_REF)
    add(queues, SubmissionRef.make("org/next-policy", "7" * 40))
    for _ in range(2):  # taken, then resumed: kept both times, and the next entry never taken
        assert daemon(duel_spec, store, tmp_path, Down(duel_spec)).step(TRACK) is False
        state = queues[TRACK].reload()
        assert state.in_progress is not None and len(state.entries) == 1
    assert daemon(duel_spec, store, tmp_path).step(TRACK) is True
    assert [r["kind"] for r in store.iter_index(TRACK)] == ["genesis", "duel"]
    assert queues[TRACK].reload().in_progress is None


def test_an_entry_whose_run_directory_holds_another_request_still_duels(
    duel_spec, store, queues, tmp_path
):
    """The same pair asked for at another size, at the block this entry gets, left a request
    behind (a kill between writing it and taking the entry). The track must not stall on it."""
    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    loop = daemon(duel_spec, store, tmp_path)
    left = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "light", block=2)
    loop.orchestrator.record_request(left)
    add(queues, REPLAY_REF, size="smoke")
    assert loop.step(TRACK) is True
    records = store.iter_index(TRACK)
    assert [(r["kind"], r["duel_size"]) for r in records[1:]] == [("duel", "smoke")]
    stale = loop.orchestrator.run_dir(left)
    assert (
        json.loads((stale.with_name(stale.name + ".stale-1") / "request.json").read_text())["size"]
        == "light"
    )
    assert queues[TRACK].reload().in_progress is None


def test_a_void_duel_finishes_its_entry_and_the_king_keeps_the_crown(
    duel_spec, store, queues, tmp_path
):
    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, REPLAY_REF)
    units = plugin_units(duel_spec, TRACK, "0" * 64, "smoke")
    lost = harness_voiding({u["unit_id"] for u in units})  # the unit ids do not depend on the id
    loop = daemon(duel_spec, store, tmp_path, resolve=lambda name: lost)
    assert loop.step(TRACK) is True
    state = queues[TRACK].reload()
    assert state.entries == [] and state.in_progress is None
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict() and len(store.iter_index(TRACK)) == 1


def test_a_king_whose_repository_is_gone_forfeits_and_the_challenger_is_crowned(
    duel_spec, store, queues, tmp_path
):
    """A king cannot hold the track by taking its repository private: it forfeits, and each
    challenger's entry buys a duel it can win."""
    from vector_orchestrator.duel.runtime import SubmissionRefused

    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, REPLAY_REF)

    class Private(FakePolicyRuntime):
        def fetch(self, ref, *, workdir):
            if ref == ZERO_REF:
                raise SubmissionRefused("resolve", "404 Client Error. Repository Not Found")
            return super().fetch(ref, workdir=workdir)

    runtime = Private(duel_spec)
    assert daemon(duel_spec, store, tmp_path, runtime).step(TRACK) is True
    records = store.iter_index(TRACK)
    assert [r["kind"] for r in records] == ["genesis", "duel"] and records[1]["dethroned"]
    assert store.head(TRACK)["king"] == REPLAY_REF.as_dict()
    event = store.event(TRACK, records[1]["event_id"])
    assert "king forfeit: resolve: 404 Client Error. Repository Not Found" in event["notes"]
    assert runtime.prepared == [REPLAY_REF.repo]
    state = queues[TRACK].reload()
    assert state.entries == [] and state.in_progress is None


def test_the_king_queued_again_is_dropped_without_a_duel(duel_spec, store, queues, tmp_path):
    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, ZERO_REF)
    runtime = FakePolicyRuntime(duel_spec)
    assert daemon(duel_spec, store, tmp_path, runtime).step(TRACK) is True
    assert queues[TRACK].entries() == [] and runtime.prepared == []
    assert len(store.iter_index(TRACK)) == 1


def test_a_track_whose_benchmark_is_missing_is_skipped_and_its_queue_kept(
    spec_doc, write_spec, store, queues, tmp_path
):
    doc = fake_spec_doc(spec_doc, pin={**FAKE_PIN, "version": "9.9.9"})
    spec = write_spec(doc, name="wrong-pin.json")
    add(queues, REPLAY_REF)
    assert daemon(spec, store, tmp_path).step(TRACK) is False
    assert [e.key for e in queues[TRACK].entries()] == [REPLAY_REF.key]


def test_a_baseline_takes_the_empty_throne_before_any_entry(
    spec_doc, write_spec, store, queues, tmp_path
):
    doc = fake_spec_doc(spec_doc)
    doc["budgets"]["act_timeout_s"] = 2.0
    doc["baselines"][TRACK] = {"repo": ZERO_REF.repo, "revision": ZERO_REF.revision}
    spec = write_spec(doc, name="baseline.json")
    add(queues, REPLAY_REF)
    loop = daemon(spec, store, tmp_path)
    assert loop.step(TRACK) is True
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict()
    assert [e.key for e in queues[TRACK].entries()] == [REPLAY_REF.key]
    assert loop.step(TRACK) is True and store.head(TRACK)["king"] == REPLAY_REF.as_dict()


class Stop(BaseException):
    """Ends a daemon's loop from inside a test."""


def crashing_steps(loop, script):
    """`step_all` playing `script`: an exception is raised, anything else returned."""
    steps = iter(script)

    def step_all():
        step = next(steps)
        if isinstance(step, BaseException):
            raise step
        return step

    loop.step_all = step_all


def test_a_step_that_keeps_crashing_is_retried_with_an_exponential_backoff(
    duel_spec, store, tmp_path
):
    slept = []
    base = daemon(duel_spec, store, tmp_path)
    loop = Daemon(base.orchestrator, base.queues, idle_sleep_s=15, sleep=slept.append)
    disk_full = OSError(28, "No space left on device")
    crashing_steps(loop, [disk_full] * 10 + [True, False, disk_full, Stop()])
    with pytest.raises(Stop):
        loop.run()
    capped = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]
    assert slept == [*capped, 15, 1], "a crash was retried at once, or success did not reset"

    slept.clear()
    loop = Daemon(base.orchestrator, base.queues, max_backoff_s=5, sleep=slept.append)
    crashing_steps(loop, [disk_full] * 5 + [Stop()])
    with pytest.raises(Stop):
        loop.run()
    assert slept == [1, 2, 4, 5, 5]


def test_the_daemon_command_takes_its_backoff_cap():
    from vector_orchestrator.cli import build_parser

    args = build_parser().parse_args(["daemon", "--store", "s", "--run-dir", "r"])
    assert args.max_backoff == 300.0
    args = build_parser().parse_args(
        ["daemon", "--store", "s", "--run-dir", "r", "--max-backoff", "7"]
    )
    assert args.max_backoff == 7.0


def test_one_daemon_per_store(duel_spec, store, tmp_path):
    with store_lock(store.root):
        with pytest.raises(RuntimeError, match="another orchestrator is publishing"):
            daemon(duel_spec, store, tmp_path).run(once=True)


def test_what_serves_beside_the_loop_starts_once_the_store_is_held(
    duel_spec, store, queues, tmp_path
):
    """The intake starts once its daemon holds the store and has published the queues, never beside
    another daemon, and never for a daemon that could not take its store."""
    add(queues, ZERO_REF)
    started: list[list[str]] = []

    def serving() -> None:
        with pytest.raises(RuntimeError, match="another orchestrator is publishing"):
            with store_lock(store.root):
                pass
        snapshot = json.loads((store.root / "tracks" / TRACK / "queue.json").read_text())
        started.append([e["key"] for e in snapshot["entries"]])

    with store_lock(store.root):
        with pytest.raises(RuntimeError, match="another orchestrator is publishing"):
            daemon(duel_spec, store, tmp_path).run(once=True, serving=serving)
    assert started == [], "it started beside another daemon"
    daemon(duel_spec, store, tmp_path).run(once=True, serving=serving)
    assert started == [[ZERO_REF.key]]


class RecordingMirror:
    """A mirror that keeps what it was asked to push."""

    def __init__(self) -> None:
        self.pushed: list[list[str]] = []

    def push(self, files, message="publish"):
        self.pushed.append(list(files))


def test_a_snapshot_published_without_the_mirror_goes_with_the_next_push(
    duel_spec, store, queues, tmp_path
):
    """The intake publishes from its request threads, which must not wait on a push to the Hub: the
    snapshot is written at once, and the daemon's next push carries it."""
    mirror = RecordingMirror()
    served = daemon(duel_spec, store, tmp_path, mirror=mirror)
    add(queues, ZERO_REF)
    served.publish_queue(TRACK, mirror=False)
    snapshot = json.loads((store.root / "tracks" / TRACK / "queue.json").read_text())
    assert [e["key"] for e in snapshot["entries"]] == [ZERO_REF.key] and mirror.pushed == []
    served.publish_queue(TRACK)
    assert mirror.pushed == [[f"tracks/{TRACK}/queue.json"]]


def test_a_snapshot_computed_first_is_never_written_over_a_newer_one(
    duel_spec, store, tmp_path, monkeypatch
):
    """The daemon and the intake's threads publish the same track: a snapshot computed before the
    intake queued and published, and written after, would leave the new entry unpublished."""
    served = daemon(duel_spec, store, tmp_path)
    write = Store.write_queue
    intake: list[threading.Thread] = []

    def queue_and_publish() -> None:
        served.queues[TRACK].offer(REPLAY_REF.repo, REPLAY_REF.revision)
        served.publish_queue(TRACK, mirror=False)

    def write_while_the_intake_publishes(self, track, snapshot):
        if not intake:
            intake.append(threading.Thread(target=queue_and_publish))
            intake[0].start()
            intake[0].join(0.3)  # an intake that does not wait has published by now
        write(self, track, snapshot)

    monkeypatch.setattr(Store, "write_queue", write_while_the_intake_publishes)
    served.publish_queue(TRACK, mirror=False)
    intake[0].join(5)
    snapshot = json.loads((store.root / "tracks" / TRACK / "queue.json").read_text())
    assert [e["key"] for e in snapshot["entries"]] == [REPLAY_REF.key]
