"""A duel from request to published record: the fake benchmark, the example policies, a real store."""

from __future__ import annotations

import json

import pytest

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
from icil_orchestrator.benchmarks.units import plugin_units
from icil_orchestrator.canon import Signer
from icil_orchestrator.duel.orchestrate import (
    OUTCOME_FILE,
    CrownMoved,
    DuelRequest,
    HarnessUnavailable,
    Orchestrator,
    side_share,
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
#: A king whose repository has gone.
GONE_REF = SubmissionRef.make("org/gone-policy", "5" * 40)


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
    clips = {p["unit_id"]: p["demo_video"] for p in event["prompts"]}
    assert clips == {u["unit_id"]: u["demo_video"] for u in event["units"]}
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


def test_an_event_names_no_demonstration_clip_the_store_does_not_hold(
    spec_doc, write_spec, store, tmp_path
):
    from conftest import fake_spec_doc

    doc = fake_spec_doc(spec_doc)
    doc["budgets"]["act_timeout_s"] = 2.0
    doc["media"]["demo_video"] = False
    spec = write_spec(doc, name="no-demo-clips.json")
    store.spec = spec
    crowned(store, spec, ZERO_REF)
    result = orchestrator(spec, store, tmp_path).run(
        DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    )
    assert result.published, result.reason
    event = event_of(store, result.record)
    assert [p["demo_video"] for p in event["prompts"]] == [None, None, None]
    assert all(p["sha256"] for p in event["prompts"])
    assert all(u["demo_video"] is None for u in event["units"])
    verified(store, spec)


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


def test_a_published_unit_names_the_scene_seed_its_prompt_was_built_on(duel_spec, store, tmp_path):
    """A unit derived with candidate seeds and no scene yet, as RoboTwin derives them: the event
    names the candidate the expert kept, which is the scene both sides played."""
    import icil_fake_benchmark

    class Candidates(icil_fake_benchmark.FakeBenchmark):
        def derive_units(self, **kwargs):
            units = super().derive_units(**kwargs)
            for unit in units:
                seed = unit["instance_params"]["scene_seed"]
                unit["instance_params"].update(scene_seed=None, scene_seeds=[seed, seed + 1])
                unit["fake_materialize"] = "reject_first"
            return units

    benchmark = Candidates()
    duel = orchestrator(duel_spec, store, tmp_path, resolve=lambda name: benchmark)
    result = duel.run(DuelRequest(TRACK, REPLAY_REF, None, "smoke", block=1))
    assert result.published, result.reason
    event = event_of(store, result.record)
    derived = plugin_units(duel_spec, TRACK, result.duel_id, "smoke", resolve=duel.resolve)
    assert len(event["units"]) == len(derived) == 3
    for unit, plan in zip(event["units"], derived, strict=True):
        candidates = plan["instance_params"]["scene_seeds"]
        assert plan["instance_params"]["scene_seed"] is None
        assert unit["instance_params"] == {**plan["instance_params"], "scene_seed": candidates[1]}
        assert unit["king_success"] is True, unit["king_error"]
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

    assert result.status == "void" and "1 of 3 units are void after the challenger" in result.reason
    first, second, third = result.units
    assert not first["void"] and second["void"] and not third["void"]
    assert "the simulator lost it" in second["challenger_error"]
    assert third["challenger_success"] is None, "a unit was played for a duel already void"
    assert [repo for repo, _ in runtime.serves] == [REPLAY_REF.repo] * 2, "the king was played"
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

    class Recording:
        def __init__(self):
            self.pushed: list[str] = []

        def push(self, files):
            self.pushed.extend(files)

    store.drain_touched()
    with pytest.raises(Crash):
        orchestrator(duel_spec, store, tmp_path, mirror=DiesOnPublish()).run(req)
    assert len(store.iter_index(TRACK)) == 2
    # A restarted process: nothing of what the killed one meant to push is in memory.
    restarted = Store(store.root, duel_spec, store.signer)
    fresh, mirror = FakePolicyRuntime(duel_spec), Recording()
    result = orchestrator(duel_spec, restarted, tmp_path, fresh, mirror=mirror).run(req)
    assert result.published and result.record["seq"] == 2 and result.reason == "margin-met"
    assert len(store.iter_index(TRACK)) == 2
    assert fresh.prepared == [] and fresh.serves == [], "a published duel was run again"
    eid = req.event_id(duel_spec)
    assert {f"events/{TRACK}/{eid}.json", f"tracks/{TRACK}/index-0000.jsonl"} <= set(mirror.pushed)
    assert f"tracks/{TRACK}/head.json" in mirror.pushed
    assert sum(f.startswith("media/") for f in mirror.pushed) == 9, (
        "the duel's clips were not pushed"
    )
    verified(store, duel_spec)


def test_a_duel_killed_between_its_index_line_and_its_head_heals_the_head_when_resumed(
    duel_spec, store, tmp_path, monkeypatch
):
    crowned(store, duel_spec, ZERO_REF)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    write_head = store.write_head

    def killed(track, **fields):
        if fields["event_id"] == req.event_id(duel_spec):
            raise Crash("killed between the index line and the head")
        return write_head(track, **fields)

    monkeypatch.setattr(store, "write_head", killed)
    with pytest.raises(Crash):
        orchestrator(duel_spec, store, tmp_path).run(req)
    assert len(store.iter_index(TRACK)) == 2 and store.head(TRACK)["king"] == ZERO_REF.as_dict()

    restarted = Store(store.root, duel_spec, store.signer)
    result = orchestrator(duel_spec, restarted, tmp_path).run(req)
    assert result.published and result.record["dethroned"] is True
    head = restarted.head(TRACK)
    assert (head["seq"], head["event_id"]) == (2, req.event_id(duel_spec))
    assert head["king"] == REPLAY_REF.as_dict(), "the dethroned king still holds the head"
    verified(restarted, duel_spec)


def test_a_duel_resumed_after_its_king_lost_the_crown_is_moved_aside_unpublished(
    duel_spec, store, tmp_path
):
    crowned(store, duel_spec, ZERO_REF)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    with pytest.raises(Crash):
        orchestrator(
            duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, crash_on_serve=4)
        ).run(req)
    # While it was stopped, someone else took ZERO's crown.
    publish(store, duel_spec, make_record(duel_spec, "duel", 3, ZERO_REF, BOMB_REF, dethroned=True))

    fresh = FakePolicyRuntime(duel_spec)
    duel = orchestrator(duel_spec, store, tmp_path, fresh)
    run_dir = duel.run_dir(req)
    with pytest.raises(CrownMoved) as moved:
        duel.run(req)
    assert moved.value.moved_to == run_dir.with_name(run_dir.name + ".stale-1")
    assert (moved.value.moved_to / "challenger" / "results.jsonl").is_file()
    assert not run_dir.exists() and fresh.prepared == [] and fresh.serves == []
    assert [r["kind"] for r in store.iter_index(TRACK)] == ["genesis", "duel"]
    assert store.head(TRACK)["king"] == BOMB_REF.as_dict()


def test_a_duel_is_not_published_against_a_king_crowned_away_while_it_ran(
    duel_spec, store, tmp_path
):
    crowned(store, duel_spec, ZERO_REF)

    class Usurped(FakePolicyRuntime):
        def serve(self, prepared, *, workdir):
            if len(self.serves) == 5:  # the king's last unit: another writer crowns BOMB
                record = make_record(duel_spec, "duel", 3, ZERO_REF, BOMB_REF, dethroned=True)
                publish(store, duel_spec, record)
            return super().serve(prepared, workdir=workdir)

    live = RecordingReporter(duel_spec)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    duel = orchestrator(duel_spec, store, tmp_path, Usurped(duel_spec), live=live)
    with pytest.raises(CrownMoved, match="the crown moved from robotensor/icil-zero-policy@"):
        duel.run(req)
    records = store.iter_index(TRACK)
    assert [(r["kind"], r["block"]) for r in records] == [("genesis", 1), ("duel", 3)]
    assert store.head(TRACK)["king"] == BOMB_REF.as_dict()
    assert not duel.run_dir(req).exists()
    assert live.frames[-1]["phase"] == "failed" and live.frames[-1]["message"].startswith("stale: ")


def test_a_refused_challenger_is_refused_and_the_king_keeps_the_crown(duel_spec, store, tmp_path):
    crowned(store, duel_spec, ZERO_REF)
    broken = write_policy_repo(tmp_path / "broken", policy="pkg.policy:Missing")
    ref = SubmissionRef.make("org/broken", "3" * 40)
    runtime = FakePolicyRuntime(duel_spec, {ref.repo: broken, ZERO_REF.repo: tmp_path})
    req = DuelRequest(TRACK, ref, ZERO_REF, "smoke", block=2)
    live = RecordingReporter(duel_spec)
    duel = orchestrator(duel_spec, store, tmp_path, runtime, live=live)
    result = duel.run(req)
    assert result.status == "refused" and not result.published
    assert result.reason.startswith("the challenger's submission was refused: hello: ")
    assert all("refused" in u["challenger_error"] for u in result.units)
    assert runtime.prepared == [ref.repo] and runtime.serves == []
    assert not (duel.run_dir(req) / "prompts").exists(), "prompts were made for a refused duel"
    assert store.head(TRACK)["king"] == ZERO_REF.as_dict() and len(store.iter_index(TRACK)) == 1
    outcome = json.loads((duel.run_dir(req) / OUTCOME_FILE).read_text())
    assert (outcome["status"], outcome["reason"]) == ("refused", result.reason)
    assert live.frames[-1]["phase"] == "failed"
    assert live.frames[-1]["message"].startswith("refused: the challenger's submission")


@pytest.mark.parametrize(
    "challenger, crowned_after", [(REPLAY_REF, REPLAY_REF), (ZERO_REF, GONE_REF)]
)
def test_a_refused_king_forfeits_every_unit_and_the_duel_is_published(
    duel_spec, store, tmp_path, challenger, crowned_after
):
    crowned(store, duel_spec, GONE_REF)
    (tmp_path / "gone").mkdir()  # the king's repository no longer holds a manifest
    runtime = FakePolicyRuntime(
        duel_spec, {REPLAY_REF.repo: REPLAY, ZERO_REF.repo: ZERO, GONE_REF.repo: tmp_path / "gone"}
    )
    req = DuelRequest(TRACK, challenger, GONE_REF, "smoke", block=2)
    duel = orchestrator(duel_spec, store, tmp_path, runtime)
    result = duel.run(req)

    assert result.published and result.kind == "duel", result.reason
    record = result.record
    assert record["dethroned"] is (crowned_after == challenger)
    assert record["king_scores"]["average"] == 0.0 and record["void"] == 0
    for unit in result.units:
        assert (unit["king_success"], unit["void"]) == (False, False)
        assert unit["king_error"].startswith("the king's submission was refused: manifest: ")
    assert [repo for repo, _ in runtime.serves] == [challenger.repo] * 3, "the king was served"
    event = event_of(store, record)
    forfeits = [n for n in event["notes"] if n.startswith("king forfeit: ")]
    assert forfeits == [f"king forfeit: {event['sides']['king']['refused']}"]
    assert forfeits[0].startswith("king forfeit: manifest: ")
    assert store.head(TRACK)["king"] == crowned_after.as_dict()
    verified(store, duel_spec)


def test_a_resumed_duel_keeps_the_kings_forfeit(duel_spec, store, tmp_path):
    crowned(store, duel_spec, GONE_REF)
    (tmp_path / "gone").mkdir()
    local = {REPLAY_REF.repo: REPLAY, GONE_REF.repo: tmp_path / "gone"}
    req = DuelRequest(TRACK, REPLAY_REF, GONE_REF, "smoke", block=2)
    with pytest.raises(Crash):
        orchestrator(
            duel_spec, store, tmp_path, FakePolicyRuntime(duel_spec, local, crash_on_serve=1)
        ).run(req)
    # The king's repository is back; the duel it forfeited is still the duel it forfeited.
    back = FakePolicyRuntime(duel_spec, {**local, GONE_REF.repo: ZERO})
    result = orchestrator(duel_spec, store, tmp_path, back).run(req)
    assert result.published and result.record["dethroned"] is True
    assert back.prepared == [REPLAY_REF.repo], "the forfeited king was asked again"
    assert all(
        u["king_error"].startswith("the king's submission was refused") for u in result.units
    )


def test_too_many_void_prompts_void_the_duel_before_either_side_runs(duel_spec, store, tmp_path):
    import icil_fake_benchmark

    experts = []

    class Broken(icil_fake_benchmark.FakeBenchmark):
        def materialize_command(self, *, unit, out_dir):
            experts.append(unit["unit_id"])
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
    assert len(experts) == 2, "an expert ran for a duel already certain to be void"


def test_a_harness_that_cannot_fetch_fails_the_duel_without_deciding_it(duel_spec, store, tmp_path):
    crowned(store, duel_spec, ZERO_REF)

    class NoHub(FakePolicyRuntime):
        def fetch(self, ref, *, workdir):
            raise RuntimeUnavailable("the Hub is unreachable")

    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    live = RecordingReporter(duel_spec)
    duel = orchestrator(duel_spec, store, tmp_path, NoHub(duel_spec), live=live)
    with pytest.raises(HarnessUnavailable, match="fetching the challenger: the Hub is unreachable"):
        duel.run(req)
    assert not (duel.run_dir(req) / OUTCOME_FILE).exists()
    assert "the Hub is unreachable" in (duel.run_dir(req) / "failed.txt").read_text()
    assert live.phases == ["fetching", "failed"]
    result = orchestrator(duel_spec, store, tmp_path).run(req)
    assert result.published, "a failed duel is not decided, and runs again"
    assert not (duel.run_dir(req) / "failed.txt").exists()


def test_a_run_directory_holding_another_duel_is_moved_aside_and_the_duel_runs(
    duel_spec, store, tmp_path
):
    crowned(store, duel_spec, ZERO_REF)
    duel = orchestrator(duel_spec, store, tmp_path)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    duel.record_request(req)
    path = duel.run_dir(req) / "request.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps({**doc, "size": "heavy"}))
    result = duel.run(req)
    assert result.published and result.record["duel_size"] == "smoke"
    stale = duel.run_dir(req).with_name(duel.run_dir(req).name + ".stale-1")
    assert json.loads((stale / "request.json").read_text())["size"] == "heavy"
    assert duel.run(req).published, "a request that matches is not moved aside"


def test_each_side_may_spend_an_even_share_of_what_materializing_left_of_the_duel():
    budgets = {"duel_wall_seconds": 28800, "side_wall_seconds": 10800}
    assert side_share(budgets, 0, 2) == 10800
    assert side_share(budgets, 18000, 2) == 5400, "the challenger could spend the king's time"
    assert side_share(budgets, 18000, 1) == 10800
    assert side_share(budgets, 30000, 2) == 0.0


def test_the_king_is_given_the_challengers_share_of_the_duel(
    spec_doc, write_spec, store, tmp_path, monkeypatch
):
    from conftest import fake_spec_doc
    from icil_orchestrator.duel import orchestrate

    doc = fake_spec_doc(spec_doc)
    doc["budgets"]["act_timeout_s"] = 2.0
    doc["budgets"]["side_wall_seconds"] = doc["budgets"]["duel_wall_seconds"]  # more than half
    spec = write_spec(doc, name="long-sides.json")
    store.spec = spec
    crowned(store, spec, ZERO_REF)
    shares = {}
    run_side = orchestrate.run_side

    def spying(spec, **kwargs):
        shares[kwargs["side"]] = kwargs["budget_s"]
        return run_side(spec, **kwargs)

    monkeypatch.setattr(orchestrate, "run_side", spying)
    result = orchestrator(spec, store, tmp_path).run(
        DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    )
    assert result.published, result.reason
    half = spec.budgets["duel_wall_seconds"] / 2
    assert shares["challenger"] == shares["king"] and half - 60 < shares["king"] <= half


def test_a_duels_identity_is_its_spec_track_and_refs():
    from icil_orchestrator.spec import load_spec

    spec = load_spec()
    a = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    assert a.duel_id(spec) == DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "heavy", 9).duel_id(spec)
    assert a.event_id(spec) != DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", 3).event_id(spec)
    assert a.duel_id(spec) != DuelRequest(TRACK, ZERO_REF, REPLAY_REF, "smoke", 2).duel_id(spec)
    assert DuelRequest.from_dict(a.as_dict(spec)) == a
    assert DuelRequest(TRACK, REPLAY_REF, None, None, 1).kind == "genesis"


@pytest.mark.parametrize("workers", [2, 4])
def test_units_played_in_parallel_publish_what_one_at_a_time_publishes(
    duel_spec, store, tmp_path, workers
):
    crowned(store, duel_spec, ZERO_REF)
    duel = orchestrator(duel_spec, store, tmp_path, workers=workers)
    req = DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2)
    result = duel.run(req)
    assert result.published and result.record["dethroned"] is True, result.reason
    assert (result.record["wins"], result.record["losses"], result.record["void"]) == (3, 0, 0)
    run_dir = duel.run_dir(req)
    for side, success in (("challenger", True), ("king", False)):
        records = read_results(run_dir / side)
        assert sorted(records) == sorted(u["unit_id"] for u in result.units)
        assert all(r["success"] is success and not r["void"] for r in records.values())
    event = event_of(store, result.record)
    assert [u["unit_id"] for u in event["units"]] == sorted(u["unit_id"] for u in event["units"])
    verified(store, duel_spec)
