"""What a killed orchestrator left running is found by its ledgers and ended, and nothing else."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from icil_orchestrator.benchmarks.subprocess_runner import run_argv
from icil_orchestrator.duel import orphans
from icil_orchestrator.duel.orphans import Ledger, boot_id, reap_ledger, reap_run_root
from icil_orchestrator.store.writer import atomic_write_json

TRACK = "franka_1arm"


def started(argv: list[str], **kwargs) -> subprocess.Popen:
    return subprocess.Popen(argv, start_new_session=True, **kwargs)


def members_soon(pgid: int, count: int) -> None:
    deadline = time.monotonic() + 10
    while len(orphans._members(pgid)) < count and time.monotonic() < deadline:
        time.sleep(0.02)


def test_a_group_a_killed_orchestrator_left_is_reaped_whole(tmp_path):
    unit = tmp_path / "runs" / TRACK / ("d" * 16) / "challenger" / "fp-000"
    unit.mkdir(parents=True)
    leader = started(["sh", "-c", "sleep 60 & sleep 60; wait"])
    try:
        Ledger(unit).started(leader.pid)
        members_soon(leader.pid, 3)
        assert len(orphans._members(leader.pid)) >= 2
        assert reap_run_root(tmp_path / "runs") == [leader.pid]
        assert orphans._members(leader.pid) == [], "a member of the group outlived the reaper"
        assert not (unit / "pids.json").exists()
    finally:
        leader.kill()
        leader.wait()


def test_a_group_whose_leader_is_gone_is_reaped_by_its_members(tmp_path):
    leader = started(["sh", "-c", "sleep 60 & echo $!"], stdout=subprocess.PIPE)
    Ledger(tmp_path).started(leader.pid)
    member = int(leader.stdout.readline())
    leader.wait()
    leader.stdout.close()
    try:
        assert orphans._members(leader.pid) == [member]
        assert reap_ledger(tmp_path) == [leader.pid]
        assert orphans._members(leader.pid) == []
    finally:
        try:
            os.kill(member, 9)
        except OSError:
            pass


def test_a_group_that_is_no_longer_the_one_recorded_is_left_alone(tmp_path):
    other = started(["sleep", "60"])
    try:
        start = orphans._start(other.pid)
        for recorded in (
            {"pgid": other.pid, "start": "1", "boot": boot_id()},  # the pid was handed on
            {"pgid": other.pid, "start": start, "boot": "another boot"},
            {"pgid": other.pid, "start": start, "boot": ""},
        ):
            atomic_write_json(tmp_path / "pids.json", [recorded])
            assert reap_ledger(tmp_path) == [] and other.poll() is None, recorded
    finally:
        other.kill()
        other.wait()


def test_a_decided_duel_is_not_searched(tmp_path):
    duel = tmp_path / "runs" / TRACK / ("e" * 16)
    (duel / "king" / "fs-001").mkdir(parents=True)
    (duel / "outcome.json").write_text("{}")
    other = started(["sleep", "60"])
    try:
        Ledger(duel / "king" / "fs-001").started(other.pid)
        assert reap_run_root(tmp_path / "runs") == [] and other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_a_ledger_names_a_group_only_while_it_runs(tmp_path):
    seen, ledger_file = tmp_path / "seen.json", tmp_path / "pids.json"
    done = run_argv(
        [
            "sh",
            "-c",
            f"while [ ! -f {ledger_file} ]; do sleep 0.01; done; cat {ledger_file} > {seen}",
        ],
        env={"PATH": os.environ["PATH"]},
        timeout_s=30,
        log_path=tmp_path / "log",
        ledger=Ledger(tmp_path),
    )
    assert done.returncode == 0
    (recorded,) = json.loads(seen.read_text())
    assert recorded["boot"] == boot_id() and recorded["start"]
    assert not (tmp_path / "pids.json").exists(), "a finished group stayed in the ledger"

    ledger = Ledger(tmp_path)
    ledger.started(os.getpid())
    ledger.started(1)
    ledger.ended(os.getpid())
    assert [g["pgid"] for g in ledger.read()] == [1]
    ledger.ended(1)
    assert not Path(ledger.path).exists()
