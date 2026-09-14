"""A duel from request to published record: the fake benchmark, the example policies, a real store."""

from __future__ import annotations

import json

import pytest

from duel_helpers import (
    REPLAY,
    REPLAY_REF,
    ZERO_REF,
    Crash,
    FakePolicyRuntime,
    RecordingReporter,
    harness_voiding,
)
from icil_orchestrator.benchmarks.units import plugin_units
from icil_orchestrator.canon import Signer
from icil_orchestrator.duel.orchestrate import (
    OUTCOME_FILE,
    DuelFailed,
    DuelRequest,
    Orchestrator,
)
from icil_orchestrator.duel.runtime import RuntimeUnavailable
from icil_orchestrator.duel.side import read_results
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.live import PHASES
from icil_orchestrator.spec import load_schema
from icil_orchestrator.store.verify import Report, SchemaCheck, verify_store
from icil_orchestrator.store.writer import Store
from store_helpers import make_record, publish
from submission_helpers import write_policy_repo

TRACK = "franka_1arm"
#: The replay example under another name: a king that wins every unit it gets to play.
BOMB_REF = SubmissionRef.make("org/bomb-policy", "4" * 40)


@pytest.fixture
def store(duel_spec, tmp_path, fake_installed):
    signer = Signer.generate()
    store = Store(tmp_path / "store", duel_spec, signer)
    store.init(signer.verify_key_hex)
    return store


def crowned(store, spec, king: SubmissionRef, block: int = 1) -> None:
    """A track whose king is `king`, published without running anything."""
    publish(store, spec, make_record(spec, "genesis", block, king, None))


def orchestrator(spec, store, tmp_path, runtime=None, **kwargs):
    kwargs.setdefault("live", RecordingReporter(spec))
    return Orchestrator(
        spec, store, runtime or FakePolicyRuntime(spec), tmp_path / "runs", **kwargs
    )


def verified(store, spec):
    report = verify_store(store.root, spec, validator_key=store.signer.verify_key_hex)
    assert report.ok, report.errors
    return report


def event_of(store, record):
    return store.event(TRACK, record["event_id"])


def test_the_replay_challenger_dethrones_the_zero_king_on_identical_prompts(
    duel_spec, store, tmp_path
):
    crowned(store, duel_spec, ZERO_REF)
    live = RecordingReporter(duel_spec)
    duel = orchestrator(duel_spec, store, tmp_path, live=live)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    result = duel.run(req)

    assert result.published and result.kind == "duel", result.reason
    record = result.record
    assert record["dethroned"] is True and record["new_king"] == REPLAY_REF.as_dict()
    assert record["challenger_scores"]["average"] == 1.0 and record["king_scores"]["average"] == 0
    assert (record["wins"], record["losses"], record["void"]) == (3, 0, 0)
    assert store.head(TRACK)["king"] == REPLAY_REF.as_dict()

    event = event_of(store, record)
    run_dir = duel.run_dir(req)
    challenger = read_results(run_dir / "challenger")
    king = read_results(run_dir / "king")
    prompts = {p["unit_id"]: p["sha256"] for p in event["prompts"]}
    assert len(event["units"]) == 3 == len(prompts)
    for unit in event["units"]:
        sha = unit["prompt_sha256"]
        assert sha and sha == unit["prompt"]["sha256"] == prompts[unit["unit_id"]]
        assert (
            challenger[unit["unit_id"]]["prompt_sha256"]
            == sha
            == king[unit["unit_id"]]["prompt_sha256"]
        )
        # What the benchmark hashed as the prompt it ran from, on each side.
        for side in ("challenger", "king"):
            result_file = run_dir / side / unit["unit_id"] / "result.json"
            assert json.loads(result_file.read_text())["prompt_sha256"] == sha
        assert (unit["challenger_success"], unit["king_success"]) == (True, False)
        assert unit["outcome"] == "challenger" and unit["void"] is False
        for clip in ("demo_video", "king_video", "challenger_video"):
            assert store.has_media(unit[clip], "mp4"), clip
        for key in (
            "unit_id",
            "skill",
            "task",
            "task_label",
            "index",
            "instance",
            "instance_params",
        ):
            assert key in unit
    derived = plugin_units(duel_spec, TRACK, req.duel_id(duel_spec), "smoke", resolve=duel.resolve)
    assert [u["unit_id"] for u in event["units"]] == [u["unit_id"] for u in derived]
    assert [u["seed"] for u in event["units"]] == [u["seed"] for u in derived]

    assert event["spec_version"] == duel_spec.version
    assert event["spec_fingerprint"] == duel_spec.fingerprint
    assert event["duel_id"] == req.duel_id(duel_spec) and event["units_per_skill"] == 1
    assert event["wall_seconds"] > 0 and event["runtime"] == "local"
    assert event["benchmarks"]["fake"]["info"]["id"] == "fake"
    assert event["benchmarks"]["fake"]["pin"]["distribution"] == "icil-fake-benchmark"
    for side, ref in (("challenger", REPLAY_REF), ("king", ZERO_REF)):
        assert event["sides"][side]["commit"] == ref.revision
        assert "base_image_digest" in event["sides"][side]
    assert event["scoring"]["reason"] == "margin-met"
    verified(store, duel_spec)

    assert live.phases == [
        "fetching",
        "checking",
        "materializing",
        "evaluating",
        "publishing",
        "done",
    ]
    assert [PHASES.index(p) for p in live.phases] == sorted(PHASES.index(p) for p in live.phases)
    sides = [f["side"] for f in live.frames if f["phase"] == "evaluating"]
    assert (
        sides[0] == "challenger"
        and sides[-1] == "king"
        and "challenger" not in sides[sides.index("king") :]
    )
    report = Report()
    for frame in live.frames:
        SchemaCheck(load_schema()).check("LiveFrame", frame, "frame", report)
    assert report.errors == []
    assert live.frames[-1]["units"][0]["challenger_video"], "clips reach the live view"


def test_the_zero_challenger_does_not_take_the_replay_kings_crown(duel_spec, store, tmp_path):
    crowned(store, duel_spec, REPLAY_REF)
    result = orchestrator(duel_spec, store, tmp_path).run(
        DuelRequest(TRACK, ZERO_REF, REPLAY_REF, "smoke", block=2)
    )
    assert result.published and result.record["dethroned"] is False
    assert result.record["new_king"] is None and result.reason == "short-of-margin"
    assert store.head(TRACK)["king"] == REPLAY_REF.as_dict()
    verified(store, duel_spec)


def test_genesis_crowns_the_first_challenger_of_an_empty_track_with_its_own_scores(
    duel_spec, store, tmp_path
):
    assert store.head(TRACK)["king"] is None
    runtime = FakePolicyRuntime(duel_spec)
    live = RecordingReporter(duel_spec)
    req = DuelRequest(TRACK, ZERO_REF, None, "smoke", block=1)
    result = orchestrator(duel_spec, store, tmp_path, runtime, live=live).run(req)
    assert result.published and result.kind == "genesis"
    record = result.record
    assert record["king"] == ZERO_REF.as_dict() and record["challenger"] is None
    assert record["king_scores"]["average"] == 0.0 and record["challenger_scores"] is None
    assert record["duel_id"] == req.duel_id(duel_spec)
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict()
    assert runtime.prepared == [ZERO_REF.repo], "a genesis has no king to check"
    event = event_of(store, record)
    assert all(
        u["king_success"] is False and u["challenger_success"] is None for u in event["units"]
    )
    assert all(u["king_video"] and u["prompt_sha256"] for u in event["units"])
    assert list(event["sides"]) == ["king"]
    assert "evaluating" in live.phases and live.phases[-1] == "done"
    verified(store, duel_spec)


def test_a_king_whose_policy_dies_on_every_unit_loses_them_and_the_crown(
    duel_spec, store, tmp_path
):
    """A king cannot keep the crown by making its own policy die: each death is its failure, not
    a void that would make the duel stand still."""
    crowned(store, duel_spec, BOMB_REF)
    runtime = FakePolicyRuntime(
        duel_spec,
        {REPLAY_REF.repo: REPLAY, BOMB_REF.repo: REPLAY},
        kill_repo=BOMB_REF.repo,
    )
    duel = orchestrator(duel_spec, store, tmp_path, runtime)
    result = duel.run(DuelRequest(TRACK, REPLAY_REF, BOMB_REF, "smoke", block=2))

    assert result.published and result.record["dethroned"] is True, result.reason
    assert result.record["void"] == 0 and result.record["wins"] == 3
    for unit in result.units:
        assert (unit["challenger_success"], unit["king_success"], unit["void"]) == (
            True,
            False,
            False,
        )
        assert "killed by signal 9" in unit["king_error"]
    assert [repo for repo, _ in runtime.serves].count(BOMB_REF.repo) == 3, "a king unit unplayed"
    assert store.head(TRACK)["king"] == REPLAY_REF.as_dict()
    verified(store, duel_spec)


def test_a_duel_with_too_many_units_void_for_the_harness_is_void(duel_spec, store, tmp_path):
    crowned(store, duel_spec, ZERO_REF)
    runtime = FakePolicyRuntime(duel_spec)
    live = RecordingReporter(duel_spec)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    ids = [u["unit_id"] for u in plugin_units(duel_spec, TRACK, req.duel_id(duel_spec), "smoke")]
    lost = harness_voiding(set(ids[1:]))
    duel = orchestrator(duel_spec, store, tmp_path, runtime, live=live, resolve=lambda name: lost)
    result = duel.run(req)

    assert result.status == "void" and "2 of 3 units are void after the challenger" in result.reason
    first, second, third = result.units
    assert not first["void"] and second["void"] and third["void"]
    assert "the simulator lost it" in second["challenger_error"]
    assert [repo for repo, _ in runtime.serves] == [REPLAY_REF.repo] * 3, "the king was played"
    assert len(store.iter_index(TRACK)) == 1, "a void duel was published"
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict()
    outcome = json.loads((duel.run_dir(req) / OUTCOME_FILE).read_text())
    assert outcome["status"] == "void" and outcome["reason"] == result.reason
    assert live.phases[-1] == "failed" and live.frames[-1]["message"].startswith("void: ")
    assert duel.run(req).status == "void", "a decided duel is decided"
    verified(store, duel_spec)


def test_a_unit_void_on_one_side_is_void_for_both_and_a_duel_within_the_limit_stands(
    spec_doc, write_spec, store, tmp_path
):
    from conftest import fake_spec_doc

    doc = fake_spec_doc(spec_doc)
    doc["budgets"]["act_timeout_s"] = 2.0
    doc["duel"]["max_void_fraction"] = 0.5
    spec = write_spec(doc, name="lenient.json")
    store.spec = spec
    crowned(store, spec, ZERO_REF)
    runtime = FakePolicyRuntime(spec)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    ids = [u["unit_id"] for u in plugin_units(spec, TRACK, req.duel_id(spec), "smoke")]
    lost = harness_voiding({ids[2]})  # the challenger's last unit
    result = orchestrator(spec, store, tmp_path, runtime, resolve=lambda name: lost).run(req)
    assert result.published, result.reason
    last = result.units[2]
    assert last["void"] and last["challenger_success"] is None and last["king_success"] is None
    assert last["outcome"] == "tie" and last["king_error"].startswith("not played: void on the")
    assert [unit for _, unit in runtime.serves].count(last["unit_id"]) == 1
    record = result.record
    assert record["void"] == 1 and record["wins"] == 2
    assert record["challenger_scores"][last["skill"]] is None, "a void unit was scored"
    assert record["dethroned"] is True
    verified(store, spec)


def test_a_restarted_duel_finishes_with_each_unit_run_once_per_side(duel_spec, store, tmp_path):
    crowned(store, duel_spec, ZERO_REF)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    # Killed after the challenger's side and one of the king's units.
    with pytest.raises(Crash):
        orchestrator(
            duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, crash_on_serve=4)
        ).run(req)
    assert len(store.iter_index(TRACK)) == 1

    fresh = FakePolicyRuntime(duel_spec)
    duel = orchestrator(duel_spec, store, tmp_path, fresh)
    result = duel.run(req)
    assert result.published and result.record["dethroned"]
    assert [unit for _, unit in fresh.serves] == [u["unit_id"] for u in result.units[1:]]
    for side in ("challenger", "king"):
        for unit in result.units:
            runs = duel.run_dir(req) / side / unit["unit_id"] / "runs.log"
            assert runs.read_text().count("\n") == 1, f"{side} {unit['unit_id']}"
    prompts = duel.run_dir(req) / "prompts" / "manifest.jsonl"
    assert len(prompts.read_text().splitlines()) == 3, "a prompt was materialized twice"
    assert len(store.iter_index(TRACK)) == 2
    verified(store, duel_spec)


def test_a_duel_killed_after_it_published_does_not_publish_twice(duel_spec, store, tmp_path):
    crowned(store, duel_spec, ZERO_REF)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)

    class DiesOnPublish:
        def push(self, files):
            if any("/index-" in f for f in files):
                raise Crash("killed right after appending the record")

    store.drain_touched()
    with pytest.raises(Crash):
        orchestrator(duel_spec, store, tmp_path, mirror=DiesOnPublish()).run(req)
    assert len(store.iter_index(TRACK)) == 2
    result = orchestrator(duel_spec, store, tmp_path).run(req)
    assert result.published and result.record["seq"] == 2
    assert len(store.iter_index(TRACK)) == 2
    verified(store, duel_spec)


def test_a_refused_challenger_voids_the_duel_and_the_king_keeps_the_crown(
    duel_spec, store, tmp_path
):
    crowned(store, duel_spec, ZERO_REF)
    broken = write_policy_repo(tmp_path / "broken", policy="pkg.policy:Missing")
    ref = SubmissionRef.make("org/broken", "3" * 40)
    runtime = FakePolicyRuntime(duel_spec, {ref.repo: broken, ZERO_REF.repo: tmp_path})
    req = DuelRequest(TRACK, ref, ZERO_REF, "smoke", block=2)
    duel = orchestrator(duel_spec, store, tmp_path, runtime)
    result = duel.run(req)
    assert result.status == "void"
    assert result.reason.startswith("the challenger's submission was refused: hello: ")
    assert all(u["void"] and "refused" in u["challenger_error"] for u in result.units)
    assert runtime.prepared == [ref.repo] and runtime.serves == []
    assert not (duel.run_dir(req) / "prompts").exists(), "prompts were made for a refused duel"
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict() and len(store.iter_index(TRACK)) == 1


def test_a_refused_king_voids_the_duel_and_keeps_the_crown(duel_spec, store, tmp_path):
    crowned(store, duel_spec, ZERO_REF)
    (tmp_path / "gone").mkdir()  # the king's repository no longer holds a manifest
    runtime = FakePolicyRuntime(
        duel_spec, {REPLAY_REF.repo: REPLAY, ZERO_REF.repo: tmp_path / "gone"}
    )
    result = orchestrator(duel_spec, store, tmp_path, runtime).run(
        DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    )
    assert result.status == "void"
    assert result.reason.startswith("the king's submission was refused: manifest: ")
    assert all(u["void"] and "refused" in u["king_error"] for u in result.units)
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict() and runtime.serves == []
    assert len(store.iter_index(TRACK)) == 1, "a refused king published a vacancy"


def test_too_many_void_prompts_void_the_duel_before_either_side_runs(duel_spec, store, tmp_path):
    import icil_fake_benchmark

    class Broken(icil_fake_benchmark.FakeBenchmark):
        def materialize_command(self, *, unit, out_dir):
            if unit["unit_id"].endswith("-001"):
                unit = {**unit, "fake_materialize": "crash"}
            return super().materialize_command(unit=unit, out_dir=out_dir)

    crowned(store, duel_spec, ZERO_REF)
    runtime = FakePolicyRuntime(duel_spec)
    result = orchestrator(duel_spec, store, tmp_path, runtime, resolve=lambda name: Broken()).run(
        DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    )
    assert result.status == "void" and result.reason == "1 of 3 prompts are void"
    assert "materialize: exited 3" in result.units[1]["challenger_error"]
    assert runtime.serves == []


def test_a_harness_that_cannot_fetch_fails_the_duel_without_deciding_it(duel_spec, store, tmp_path):
    crowned(store, duel_spec, ZERO_REF)

    class NoHub(FakePolicyRuntime):
        def fetch(self, ref, *, workdir):
            raise RuntimeUnavailable("the Hub is unreachable")

    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    live = RecordingReporter(duel_spec)
    duel = orchestrator(duel_spec, store, tmp_path, NoHub(duel_spec), live=live)
    with pytest.raises(DuelFailed, match="fetching the challenger: the Hub is unreachable"):
        duel.run(req)
    assert not (duel.run_dir(req) / OUTCOME_FILE).exists()
    assert "the Hub is unreachable" in (duel.run_dir(req) / "failed.txt").read_text()
    assert live.phases == ["fetching", "failed"]
    result = orchestrator(duel_spec, store, tmp_path).run(req)
    assert result.published, "a failed duel is not decided, and runs again"
    assert not (duel.run_dir(req) / "failed.txt").exists()


def test_a_run_directory_holding_another_duel_is_refused(duel_spec, store, tmp_path):
    duel = orchestrator(duel_spec, store, tmp_path)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    duel.record_request(req)
    path = duel.run_dir(req) / "request.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps({**doc, "size": "heavy"}))
    with pytest.raises(DuelFailed, match="holds another duel's request"):
        duel.run(req)


def test_a_duels_identity_is_its_spec_track_and_refs():
    from icil_orchestrator.spec import load_spec

    spec = load_spec()
    a = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    assert a.duel_id(spec) == DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "heavy", 9).duel_id(spec)
    assert a.event_id(spec) != DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", 3).event_id(spec)
    assert a.duel_id(spec) != DuelRequest(TRACK, ZERO_REF, REPLAY_REF, "smoke", 2).duel_id(spec)
    assert DuelRequest.from_dict(a.as_dict(spec)) == a
    assert DuelRequest(TRACK, REPLAY_REF, None, None, 1).kind == "genesis"
