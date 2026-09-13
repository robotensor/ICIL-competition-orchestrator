"""The orchestrator loop over a real queue and store, with the fake benchmark and example policies."""

from __future__ import annotations

import json

import pytest

from conftest import FAKE_PIN, fake_spec_doc
from duel_helpers import REPLAY_REF, ZERO_REF, Crash, FakePolicyRuntime, RecordingReporter
from icil_orchestrator.canon import Signer
from icil_orchestrator.daemon import Daemon
from icil_orchestrator.duel.orchestrate import Orchestrator
from icil_orchestrator.queue import Queues
from icil_orchestrator.spec import load_schema
from icil_orchestrator.store.verify import Report, SchemaCheck, verify_store
from icil_orchestrator.store.writer import Store, store_lock
from store_helpers import make_record, publish

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


def daemon(spec, store, tmp_path, runtime=None):
    """A daemon as a fresh process would make one: its own store object, runtime and reporter."""
    fresh = Store(store.root, spec, store.signer)
    orchestrator = Orchestrator(
        spec,
        fresh,
        runtime or FakePolicyRuntime(spec),
        tmp_path / "runs",
        live=RecordingReporter(spec),
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


def test_a_void_duel_finishes_its_entry_and_the_king_keeps_the_crown(
    duel_spec, store, queues, tmp_path
):
    publish(store, duel_spec, make_record(duel_spec, "genesis", 1, ZERO_REF, None))
    add(queues, REPLAY_REF)
    runtime = FakePolicyRuntime(duel_spec, kill_on_serve={0})
    loop = daemon(duel_spec, store, tmp_path, runtime)
    assert loop.step(TRACK) is True
    state = queues[TRACK].reload()
    assert state.entries == [] and state.in_progress is None
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict() and len(store.iter_index(TRACK)) == 1


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


def test_one_daemon_per_store(duel_spec, store, tmp_path):
    with store_lock(store.root):
        with pytest.raises(RuntimeError, match="another orchestrator is publishing"):
            daemon(duel_spec, store, tmp_path).run(once=True)
