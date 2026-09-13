"""One side of a duel against the fake benchmark, with the example policies served per unit."""

from __future__ import annotations

import os
import time

import pytest

from duel_helpers import REPLAY_REF, ZERO_REF, Crash, FakePolicyRuntime
from icil_orchestrator.benchmarks.units import plugin_units
from icil_orchestrator.duel.materialize import materialize_units
from icil_orchestrator.duel.side import RESULTS_FILE, read_results, run_side

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


def test_a_runtime_that_dies_voids_its_unit_and_every_unit_after_it(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec, kill_on_serve={1})
    results = side(duel_spec, fake, units, prompts, tmp_path, runtime)
    first, second, third = (results[u["unit_id"]] for u in units)
    assert first["success"] is True and not first["void"]
    assert second["void"] and second["success"] is None
    assert second["error"].startswith("the challenger's policy runtime died: the policy process")
    assert third["void"] and "died earlier in this side" in third["error"]
    assert len(runtime.serves) == 2, "a dead runtime was asked to serve again"
    assert runs(tmp_path, units[2]["unit_id"]) == 0


def test_a_refused_submission_voids_every_unit_with_the_reason_and_serves_nothing(
    duel_spec, fake, units, prompts, tmp_path
):
    runtime = FakePolicyRuntime(duel_spec)
    results = side(
        duel_spec, fake, units, prompts, tmp_path, runtime, refused="manifest: icil.yaml: api"
    )
    assert all(r["void"] for r in results.values())
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
