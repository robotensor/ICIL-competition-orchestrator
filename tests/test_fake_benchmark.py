"""The fake benchmark, as a duel uses it: a real prompt, and a served policy driven through it.

The replay example copies the demonstration's actions and the zero example does not, so against
the fake benchmark one wins and the other loses - which is what a duel test needs to tell a crown
that moved from one that did not.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from icil_orchestrator.benchmarks import subprocess_runner as runner

EXAMPLES = Path(__file__).resolve().parents[1] / "packages" / "icil-policy" / "examples"
AUTHKEY_ENV = "ICIL_TEST_POLICY_AUTHKEY"


@pytest.fixture
def fake(fake_installed):
    import icil_fake_benchmark

    return icil_fake_benchmark.BENCHMARK


def a_unit(fake, **extra):
    (derived,) = fake.derive_units(
        seed_material="duel|franka_pick_and_place", count=1, suite="franka_1arm"
    )
    return {**derived, "unit_id": "fp-000", **extra}


def materialized(fake, unit, out: Path) -> Path:
    argv = fake.materialize_command(unit=unit, out_dir=str(out))
    done = runner.run_argv(argv, env=dict(os.environ), timeout_s=60, log_path=out / "m.log")
    assert done.returncode == 0, done.log_tail
    return out / "prompt.npz"


def serve(example: str, workdir: Path):
    """`python -m icil_policy.serve` on an example repository; `(process, address, key)`."""
    key = secrets.token_bytes(32)
    address = Path(tempfile.mkdtemp(prefix="icil-duel-test-")) / "policy.sock"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "icil_policy.serve",
            "--manifest",
            str(EXAMPLES / example / "icil.yaml"),
            "--address",
            str(address),
            "--authkey-env",
            AUTHKEY_ENV,
            "--log-file",
            str(workdir / "policy.log"),
        ],
        env={**os.environ, AUTHKEY_ENV: key.hex()},
        stdin=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while not address.exists():
        assert process.poll() is None and time.monotonic() < deadline, "the policy never listened"
        time.sleep(0.05)
    return process, str(address), key


def test_a_prompt_is_named_arrays_with_its_meta_and_verifies(fake, tmp_path):
    import numpy as np

    unit = a_unit(fake)
    prompt = materialized(fake, unit, tmp_path / "prompt")
    with np.load(prompt) as arrays:
        assert {"frames_head_camera", "qpos", "actions", "meta"} <= set(arrays.files)
        assert arrays["actions"].shape == (5, 16)
    verdict = fake.verify_prompt(path=str(prompt), unit=unit)
    assert verdict["ok"], verdict["problems"]
    from icil_orchestrator.canon import sha256_file

    assert verdict["sha256"] == sha256_file(prompt)


def test_a_prompt_for_another_scene_does_not_verify(fake, tmp_path):
    unit = a_unit(fake, fake_materialize="wrong")
    prompt = materialized(fake, unit, tmp_path / "prompt")
    verdict = fake.verify_prompt(path=str(prompt), unit=unit)
    assert not verdict["ok"] and "scene_seed" in verdict["problems"][0]
    assert not fake.verify_prompt(path=str(tmp_path / "missing.npz"), unit=unit)["ok"]


@pytest.mark.parametrize("example, wins", [("replay_policy", True), ("zero_policy", False)])
def test_the_replay_policy_solves_a_unit_and_the_zero_policy_does_not(
    fake, tmp_path, example, wins
):
    unit = a_unit(fake)
    prompt = materialized(fake, unit, tmp_path / "prompt")
    process, address, key = serve(example, tmp_path)
    try:
        outcome = runner.run_unit(
            fake,
            unit,
            prompt=str(prompt),
            out_dir=tmp_path / "run",
            policy_address=address,
            authkey_env=AUTHKEY_ENV,
            timeout_s=60,
            env={**runner.benchmark_environment(os.environ, AUTHKEY_ENV), AUTHKEY_ENV: key.hex()},
            extra={"act_timeout_s": 10.0},
        )
        assert process.wait(timeout=30) == 0, "the server did not end with its client"
    finally:
        process.kill()
        shutil.rmtree(Path(address).parent, ignore_errors=True)
    assert not outcome.void, outcome.error
    assert outcome.success is wins and outcome.steps == 5
    assert outcome.progress == (1.0 if wins else 0.0)
    assert outcome.clip and (tmp_path / "run" / "runs.log").read_text().count("\n") == 1


def test_a_policy_that_is_not_there_is_a_failed_episode_with_the_reason(fake, tmp_path):
    unit = a_unit(fake)
    prompt = materialized(fake, unit, tmp_path / "prompt")
    outcome = runner.run_unit(
        fake,
        unit,
        prompt=str(prompt),
        out_dir=tmp_path / "run",
        policy_address=str(tmp_path / "nobody.sock"),
        authkey_env=AUTHKEY_ENV,
        timeout_s=60,
        env={"PATH": "/usr/bin:/bin", AUTHKEY_ENV: "00" * 32},
        extra={"act_timeout_s": 0.5},
    )
    assert (outcome.success, outcome.void) == (False, False)
    assert outcome.error.startswith("policy: connect:")
