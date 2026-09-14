"""A duel stopped by a signal takes its unit's policy server and benchmark down before it exits.

These run `icil-orchestrator duel` as a real process on the local runtime, with a policy slowed
down so that a unit is surely under way, and look in /proc for what it left running.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import FAKE_SITE
from duel_helpers import REPLAY
from icil_orchestrator.cli import main
from icil_orchestrator.ids import SubmissionRef

TRACK = "franka_1arm"
SLOW_REF = SubmissionRef.make("org/slow-policy", "8" * 40)
#: Per `act`: a unit of five acts takes a few seconds, under the duel spec's 2s act timeout.
SLOW_ACT_S = 0.6


@pytest.fixture
def slow_policy(tmp_path) -> Path:
    """The replay example, sleeping in every `act`."""
    root = tmp_path / "slow"
    shutil.copytree(REPLAY, root, ignore=shutil.ignore_patterns("__pycache__"))
    policy = root / "replay" / "policy.py"
    text = policy.read_text()
    text = text.replace("import numpy as np", "import time\n\nimport numpy as np", 1)
    text = text.replace(
        "        k = min(self._step,",
        f"        time.sleep({SLOW_ACT_S})\n        k = min(self._step,",
        1,
    )
    policy.write_text(text)
    return root


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except OSError:
        return False
    return state != "Z"


def processes_running(text: str) -> list[int]:
    """The live processes whose command line mentions `text`."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if text in cmdline and alive(int(entry.name)):
            found.append(int(entry.name))
    return found


def start_duel(spec, tmp_path: Path, slow: Path) -> subprocess.Popen:
    root, key = tmp_path / "store", tmp_path / "keys" / "orchestrator.ed25519"
    assert main(["--spec", str(spec.path), "store", "init", str(root), "--key", str(key)]) == 0
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(FAKE_SITE), *sys.path])}
    argv = [
        sys.executable,
        "-m",
        "icil_orchestrator",
        "--spec",
        str(spec.path),
        "duel",
        "--challenger",
        SLOW_REF.entry,
        "--store",
        str(root),
        "--run-dir",
        str(tmp_path / "runs"),
        "--queue",
        str(tmp_path / "queue"),
        "--key",
        str(key),
        "--runtime",
        "local",
        "--local",
        f"{SLOW_REF.repo}={slow}",
    ]
    log = open(tmp_path / "duel.log", "wb")  # noqa: SIM115 - the child writes it until it exits
    return subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=tmp_path)


def mid_unit(process: subprocess.Popen, tmp_path: Path, slow: Path) -> tuple[int, list[int]]:
    """Once a unit is under way: the benchmark's pid and the unit's policy server's."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        assert process.poll() is None, (tmp_path / "duel.log").read_text()
        started = sorted((tmp_path / "runs").glob(f"{TRACK}/*/challenger/*/runs.log"))
        policies = processes_running(str(slow / "icil.yaml"))
        if started and policies and started[0].read_text().strip():
            return int(started[0].read_text().split()[0]), policies
        time.sleep(0.05)
    raise AssertionError("no unit got under way")


def reap(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_tears_down_the_units_policy_and_benchmark_before_the_duel_exits(
    duel_spec, fake_installed, tmp_path, slow_policy, signum
):
    process = start_duel(duel_spec, tmp_path, slow_policy)
    benchmark, policies = [], []
    try:
        benchmark_pid, policies = mid_unit(process, tmp_path, slow_policy)
        benchmark = [benchmark_pid]
        process.send_signal(signum)
        code = process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        left = [pid for pid in [*benchmark, *policies] if alive(pid)]
        reap(left)
    log = (tmp_path / "duel.log").read_text()
    assert left == [], f"left running: {left}\n{log}"
    assert code == 128 + signum, log
    assert f"stopped by {signal.Signals(signum).name}" in log
