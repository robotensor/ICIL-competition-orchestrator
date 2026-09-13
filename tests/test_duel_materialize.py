"""Prompts produced once per duel, through the fake benchmark's real materialize command."""

from __future__ import annotations

import json
import os

import pytest

from icil_orchestrator.benchmarks.units import plugin_units
from icil_orchestrator.canon import sha256_file
from icil_orchestrator.duel.materialize import (
    MANIFEST_FILE,
    materialize_units,
    read_manifest,
)

TRACK = "franka_1arm"


@pytest.fixture
def fake(fake_installed):
    import icil_fake_benchmark

    return icil_fake_benchmark.BENCHMARK


@pytest.fixture
def units(fake_spec, fake):
    return plugin_units(fake_spec, TRACK, "d" * 64, "smoke", resolve=lambda name: fake)


def counting(fake):
    """The fake, counting its materialize commands."""
    calls = []

    class Counted(type(fake)):
        def materialize_command(self, *, unit, out_dir):
            calls.append(unit["unit_id"])
            return super().materialize_command(unit=unit, out_dir=out_dir)

    return Counted(), calls


def test_every_unit_gets_a_verified_prompt_and_its_hash(fake_spec, fake, units, tmp_path):
    seen = []
    out = materialize_units(
        fake_spec,
        units,
        tmp_path,
        benchmark_of=lambda unit: fake,
        on_prompt=lambda unit, prompt: seen.append(unit["unit_id"]),
    )
    assert seen == [u["unit_id"] for u in units] == list(out.prompts)
    for unit in units:
        prompt = out.prompts[unit["unit_id"]]
        assert not prompt.void, prompt.error
        assert prompt.path == tmp_path / unit["unit_id"] / "prompt.npz"
        assert prompt.sha256 == sha256_file(prompt.path)
        assert prompt.demo_sha256 == sha256_file(prompt.demo_clip)
        assert prompt.path.stat().st_mode & 0o222 == 0, "a side's benchmark could rewrite it"
    assert out.manifest() == [
        {
            "unit_id": u["unit_id"],
            "sha256": out.prompts[u["unit_id"]].sha256,
            "demo_video": out.prompts[u["unit_id"]].demo_sha256,
        }
        for u in units
    ]
    assert out.void == 0


@pytest.mark.parametrize(
    "behaviour, reason",
    [
        ("crash", "exited 3: the expert lost the GPU"),
        ("wrong", "the prompt is not the one the unit asked for: scene_seed"),
        ("expert_fails", "the demonstration did not succeed: the expert never succeeded"),
    ],
)
def test_a_unit_whose_prompt_fails_is_void_with_the_reason_and_the_rest_go_on(
    fake_spec, fake, units, tmp_path, behaviour, reason
):
    units[1]["fake_materialize"] = behaviour
    out = materialize_units(fake_spec, units, tmp_path, benchmark_of=lambda unit: fake)
    bad = out.prompts[units[1]["unit_id"]]
    assert bad.void and bad.sha256 is None and bad.path is None
    assert bad.error.startswith("fake: materialize: ") and reason in bad.error
    assert [p.void for p in out.prompts.values()] == [False, True, False]
    assert [e["unit_id"] for e in out.manifest()] == [units[0]["unit_id"], units[2]["unit_id"]]


def test_a_materialization_past_its_budget_is_void(fake_spec, fake, units, tmp_path):
    units[0]["fake_materialize"] = "hang"
    out = materialize_units(
        fake_spec, units[:1], tmp_path, benchmark_of=lambda unit: fake, timeout_s=1.0
    )
    assert out.prompts[units[0]["unit_id"]].error == "fake: materialize: exceeded its 1s budget"


def test_a_verify_prompt_that_hashes_something_else_is_refused(fake_spec, fake, units, tmp_path):
    class Hashes(type(fake)):
        def verify_prompt(self, *, path, unit):
            return {**super().verify_prompt(path=path, unit=unit), "sha256": "0" * 64}

    out = materialize_units(fake_spec, units[:1], tmp_path, benchmark_of=lambda unit: Hashes())
    assert "not the sha256 of its bytes" in out.prompts[units[0]["unit_id"]].error


def test_a_restarted_duel_reuses_its_prompts_rather_than_producing_new_ones(
    fake_spec, fake, units, tmp_path
):
    counted, calls = counting(fake)
    first = materialize_units(fake_spec, units[:2], tmp_path, benchmark_of=lambda u: counted)
    assert calls == [units[0]["unit_id"], units[1]["unit_id"]]
    # A crash while the third was being written: its line never made it.
    again = materialize_units(fake_spec, units, tmp_path, benchmark_of=lambda u: counted)
    assert calls == [u["unit_id"] for u in units], "a finished prompt was produced twice"
    for unit_id, prompt in first.prompts.items():
        assert again.prompts[unit_id].sha256 == prompt.sha256
    assert set(read_manifest(tmp_path)) == {u["unit_id"] for u in units}


def test_a_prompt_changed_on_disk_is_void_on_resume_not_trusted(fake_spec, fake, units, tmp_path):
    first = materialize_units(fake_spec, units[:1], tmp_path, benchmark_of=lambda u: fake)
    path = first.prompts[units[0]["unit_id"]].path
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b"\0")
    again = materialize_units(fake_spec, units[:1], tmp_path, benchmark_of=lambda u: fake)
    prompt = again.prompts[units[0]["unit_id"]]
    assert prompt.void and "changed after it was materialized" in prompt.error
    lines = (tmp_path / MANIFEST_FILE).read_text().splitlines()
    assert json.loads(lines[-1])["void"] is True and len(lines) == 2
