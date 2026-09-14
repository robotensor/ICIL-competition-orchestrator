"""One side of a duel against the fake benchmark, with the example policies served per unit."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from duel_helpers import REPLAY_REF, ZERO_REF, Crash, FakePolicyRuntime
from icil_orchestrator.benchmarks.subprocess_runner import Outcome, voided
from icil_orchestrator.benchmarks.units import plugin_units
from icil_orchestrator.duel.materialize import materialize_units
from icil_orchestrator.duel.runtime import (
    HARNESS,
    POLICY,
    PolicyDied,
    PolicyEnd,
    RuntimeUnavailable,
)
from icil_orchestrator.duel.side import RESULTS_FILE, attribute, read_results, run_side

TRACK = "franka_1arm"


@pytest.fixture
def fake(fake_installed):
    import icil_fake_benchmark

    return icil_fake_benchmark.BENCHMARK


@pytest.fixture
def units(duel_spec, fake):
    return plugin_units(duel_spec, TRACK, "e" * 64, "smoke", resolve=lambda name: fake)


@pytest.fixture
def prompts(duel_spec, fake, units, tmp_path):
    return materialize_units(duel_spec, units, tmp_path / "prompts", benchmark_of=lambda u: fake)


def side(duel_spec, fake, units, prompts, tmp_path, runtime, ref=REPLAY_REF, **kwargs):
    prepared = kwargs.pop("prepared", None)
    if prepared is None and "refused" not in kwargs:
        fetched = runtime.fetch(ref, workdir=tmp_path / "challenger")
        prepared = runtime.prepare(fetched, workdir=tmp_path / "challenger" / "check")
    return run_side(
        duel_spec,
        side="challenger",
        units=units,
        prompts=prompts,
        side_dir=tmp_path / "challenger",
        benchmark_of=lambda unit: fake,
        runtime=runtime,
        prepared=prepared,
        **kwargs,
    )


def runs(tmp_path, unit_id):
    log = tmp_path / "challenger" / unit_id / "runs.log"
    return log.read_text().count("\n") if log.exists() else 0


@pytest.mark.parametrize("ref, wins", [(REPLAY_REF, True), (ZERO_REF, False)])
def test_every_unit_runs_against_its_own_policy_and_is_recorded(
    duel_spec, fake, units, prompts, tmp_path, ref, wins
):
    runtime = FakePolicyRuntime(duel_spec)
    seen = []
    results = side(
        duel_spec,
        fake,
        units,
        prompts,
        tmp_path,
        runtime,
        ref=ref,
        on_unit=lambda unit, record: seen.append(record["unit_id"]),
    )
    ids = [u["unit_id"] for u in units]
    assert seen == ids == list(results)
    assert [s[1] for s in runtime.serves] == ids, "one served policy per unit"
    for unit in units:
        record = results[unit["unit_id"]]
        assert (record["success"], record["void"]) == (wins, False), record["error"]
        assert record["prompt_sha256"] == prompts.prompts[unit["unit_id"]].sha256
        assert record["clip"] == f"{unit['unit_id']}/evaluation.mp4" and record["clip_sha256"]
        assert (tmp_path / "challenger" / unit["unit_id"] / "policy.log").is_file()
        assert runs(tmp_path, unit["unit_id"]) == 1
    assert read_results(tmp_path / "challenger") == results


def test_a_restarted_side_runs_only_the_units_it_had_not_finished(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec, crash_on_serve=1)
    with pytest.raises(Crash):
        side(duel_spec, fake, units, prompts, tmp_path, runtime)
    assert list(read_results(tmp_path / "challenger")) == [units[0]["unit_id"]]

    results = side(duel_spec, fake, units, prompts, tmp_path, FakePolicyRuntime(duel_spec))
    assert all(r["success"] for r in results.values())
    for unit in units:
        assert runs(tmp_path, unit["unit_id"]) == 1, f"{unit['unit_id']} ran twice or never"
    lines = (tmp_path / "challenger" / RESULTS_FILE).read_text().splitlines()
    assert len(lines) == len(units)


def test_a_policy_that_dies_under_its_unit_fails_it_and_the_next_unit_is_served_afresh(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec, kill_on_serve={1})
    results = side(duel_spec, fake, units, prompts, tmp_path, runtime)
    first, second, third = (results[u["unit_id"]] for u in units)
    assert first["success"] is True and not first["void"]
    assert (second["success"], second["void"]) == (False, False), "a policy's death voided it"
    assert "policy: " in second["error"] and "killed by signal 9" in second["error"]
    assert third["success"] is True and not third["void"]
    assert len(runtime.serves) == 3 and runs(tmp_path, units[2]["unit_id"]) == 1


@pytest.mark.parametrize(
    "behaviour, cause, outcome",
    [
        ("policy_then_void", "policy", (False, False)),
        ("policy_then_void", "harness", (None, True)),
        ("policy_then_void", "", (None, True)),
        ("crash", "", (None, True)),
    ],
)
def test_a_void_the_benchmark_reports_is_the_sides_failure_only_for_the_policys_cause(
    duel_spec, fake, units, prompts, tmp_path, behaviour, cause, outcome
):
    units[0].update(fake_behaviour=behaviour, fake_void_cause=cause)
    results = side(duel_spec, fake, units, prompts, tmp_path, FakePolicyRuntime(duel_spec))
    record = results[units[0]["unit_id"]]
    assert (record["success"], record["void"]) == outcome, record["error"]
    assert record["error"]


def test_a_policy_that_never_listens_fails_and_a_runtime_that_cannot_serve_voids(
    duel_spec, fake, units, prompts, tmp_path
):
    class Flaky(FakePolicyRuntime):
        def serve(self, prepared, *, workdir):
            unit = Path(workdir).name
            if unit == units[0]["unit_id"]:
                raise PolicyDied("the policy did not listen within 600s")
            if unit == units[1]["unit_id"]:
                raise RuntimeUnavailable("docker is not running")
            return super().serve(prepared, workdir=workdir)

    results = side(duel_spec, fake, units, prompts, tmp_path, Flaky(duel_spec))
    never, unserved, fine = (results[u["unit_id"]] for u in units)
    assert (never["success"], never["void"]) == (False, False)
    assert "did not listen" in never["error"]
    assert (unserved["success"], unserved["void"]) == (None, True)
    assert "docker is not running" in unserved["error"]
    assert fine["success"] is True


@pytest.mark.parametrize(
    "void_cause, end, outcome",
    [
        (None, None, (None, True)),
        ("harness", None, (None, True)),
        ("policy", None, (False, False)),
        ("nonsense", None, (None, True)),
        (None, PolicyEnd("the policy process exited 1", POLICY), (False, False)),
        ("harness", PolicyEnd("the policy process exited 1", POLICY), (None, True)),
        ("policy", PolicyEnd("the policy container is gone", HARNESS), (None, True)),
        (None, PolicyEnd("the policy container is gone", HARNESS), (None, True)),
    ],
)
def test_whose_a_void_is(void_cause, end, outcome):
    reported = voided("fake: policy: act: no answer within 30s")
    if void_cause is not None:
        reported.extra["void_cause"] = void_cause
    attributed = attribute(reported, end)
    assert (attributed.success, attributed.void) == outcome
    assert attributed.error.startswith("fake: policy: act")
    scored = Outcome(success=True, void=False, steps=5, error=None)
    assert attribute(scored, PolicyEnd("exited 1", POLICY)) is scored, "a score was undone"


def test_a_refused_submission_fails_every_unit_with_the_reason_and_serves_nothing(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec)
    results = side(
        duel_spec, fake, units, prompts, tmp_path, runtime, refused="manifest: icil.yaml: api"
    )
    assert all((r["success"], r["void"]) == (False, False) for r in results.values())
    assert {r["error"] for r in results.values()} == {
        "the challenger's submission was refused: manifest: icil.yaml: api"
    }
    assert runtime.serves == []


def test_a_unit_without_a_prompt_is_void_and_the_others_run(duel_spec, fake, units, tmp_path):
    units[0]["fake_materialize"] = "crash"
    prompts = materialize_units(duel_spec, units, tmp_path / "prompts", benchmark_of=lambda u: fake)
    runtime = FakePolicyRuntime(duel_spec)
    results = side(duel_spec, fake, units, prompts, tmp_path, runtime)
    bad = results[units[0]["unit_id"]]
    assert bad["void"] and bad["error"].startswith("no prompt: fake: materialize: exited 3")
    assert [r["success"] for r in results.values()] == [None, True, True]
    assert len(runtime.serves) == 2


def test_a_prompt_changed_since_it_was_materialized_is_not_run(
    duel_spec, fake, units, prompts, tmp_path
):
    path = prompts.prompts[units[1]["unit_id"]].path
    os.chmod(path, 0o644)
    path.write_bytes(b"not the prompt")
    results = side(duel_spec, fake, units, prompts, tmp_path, FakePolicyRuntime(duel_spec))
    assert "changed after it was materialized" in results[units[1]["unit_id"]]["error"]
    assert runs(tmp_path, units[1]["unit_id"]) == 0


def test_units_left_when_the_duel_runs_out_of_time_are_void(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec)
    results = side(
        duel_spec, fake, units, prompts, tmp_path, runtime, deadline=time.monotonic() - 1
    )
    assert {r["error"] for r in results.values()} == {"the side ran out of its wall-clock budget"}
    assert runtime.serves == []


def test_an_interrupted_unit_is_reaped_and_moved_aside_before_it_runs_again(
    duel_spec, fake, units, prompts, tmp_path
):
    import subprocess

    from icil_orchestrator.duel.orphans import Ledger

    first = units[0]["unit_id"]
    unit_dir = tmp_path / "challenger" / first
    unit_dir.mkdir(parents=True)
    (unit_dir / "runs.log").write_text("12345\n")
    orphan = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        Ledger(unit_dir).started(orphan.pid)  # what a killed orchestrator's benchmark left
        results = side(duel_spec, fake, units, prompts, tmp_path, FakePolicyRuntime(duel_spec))
        assert orphan.wait(timeout=10) == -9, "the orphan was not ended"
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()
    assert results[first]["success"] is True
    aside = unit_dir.with_name(f"{first}.interrupted-1")
    assert (aside / "runs.log").read_text() == "12345\n"
    assert runs(tmp_path, first) == 1 and not (unit_dir / "pids.json").exists()


def test_a_prompt_changed_while_its_unit_ran_voids_the_unit(
    duel_spec, fake, units, prompts, tmp_path
):
    target = prompts.prompts[units[1]["unit_id"]].path

    class Tampering(FakePolicyRuntime):
        def _started(self, process, served):
            super()._started(process, served)
            if served.log_file.parent.name == units[1]["unit_id"]:
                os.chmod(target, 0o644)
                target.write_bytes(target.read_bytes() + b"\0")

    results = side(duel_spec, fake, units, prompts, tmp_path, Tampering(duel_spec))
    changed = results[units[1]["unit_id"]]
    assert changed["void"] and "changed after it was materialized" in changed["error"]
    assert runs(tmp_path, units[1]["unit_id"]) == 1, "the check is after the unit, not before"
    assert results[units[0]["unit_id"]]["success"] is True


def test_a_unit_the_benchmark_says_ran_from_another_prompt_is_void(
    duel_spec, fake, units, prompts, tmp_path
):
    class Misreading(type(fake)):
        def read_result(self, *, out_dir):
            result = super().read_result(out_dir=out_dir)
            if Path(out_dir).name == units[2]["unit_id"]:
                result["prompt_sha256"] = "f" * 64
            return result

    misreading = Misreading()
    runtime = FakePolicyRuntime(duel_spec)
    results = run_side(
        duel_spec,
        side="challenger",
        units=units,
        prompts=prompts,
        side_dir=tmp_path / "challenger",
        benchmark_of=lambda unit: misreading,
        runtime=runtime,
        prepared=runtime.prepare(
            runtime.fetch(REPLAY_REF, workdir=tmp_path), workdir=tmp_path / "check"
        ),
    )
    other = results[units[2]["unit_id"]]
    assert other["void"] and "the benchmark read a prompt hashing to ffffffffffff" in other["error"]
    assert [results[u["unit_id"]]["void"] for u in units[:2]] == [False, False]


def test_a_side_given_less_than_its_budget_stops_at_its_share(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec)
    results = side(duel_spec, fake, units, prompts, tmp_path, runtime, budget_s=0)
    assert {r["error"] for r in results.values()} == {"the side ran out of its wall-clock budget"}
    assert runtime.serves == []


def test_a_unit_the_duel_already_holds_void_is_not_played(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec)
    reason = "void on the challenger's side: the policy runtime died"
    results = side(
        duel_spec, fake, units, prompts, tmp_path, runtime, void_units={units[0]["unit_id"]: reason}
    )
    assert results[units[0]["unit_id"]]["error"] == f"not played: {reason}"
    assert [s[1] for s in runtime.serves] == [u["unit_id"] for u in units[1:]]
