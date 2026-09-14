"""What a killed orchestrator left running, found and ended before a unit runs again.

`SIGTERM` and `SIGINT` tear a unit down on the way out (see `cli`), but `SIGKILL`, the OOM killer
or a lost machine run no `finally`. What survives them:

- the benchmark subprocess's process group and the local runtime's policy server, which run in
  sessions of their own and so outlive the orchestrator;
- a policy container, which Docker keeps running whoever started it.

So every process group a unit starts is written, while it runs, to a ledger in the directory it
runs in (`pids.json`: the unit's, a prompt's, a side's health check), and every policy container
carries labels naming the store and the run root it serves (`docker_runtime`). On start, holding
the store's lock so that nothing of this store's can still be running on purpose, the
orchestrator reaps both (`Orchestrator.reap_orphans`). `run_side` also reaps a unit's own ledger
before it plays a unit that has no result, and moves what that interrupted attempt left aside as
`<unit_id>.interrupted-<n>`, so nothing an orphan wrote is ever read as the unit's result.

A process group is killed only while it is still the one recorded - the same boot, and a leader
that started when the ledger says it did (field 22 of `/proc/<pid>/stat`, since a pid is reused),
or, with the leader gone, members still in that group and session: the kernel does not hand out a
process group's id again while any member holds it.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

from ..store.writer import atomic_write_json

log = logging.getLogger(__name__)

LEDGER_FILE = "pids.json"
#: How long a killed group gets to be gone before the reaper moves on.
KILL_WAIT_S = 5.0
POLL_S = 0.02


def _stat(pid: int | str) -> list[str] | None:
    """`/proc/<pid>/stat` from field 3 (the state) on; None for no such process. The command
    name before it is in parentheses and may hold anything, so the split is after its last ")"."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    fields = text.rpartition(")")[2].split()
    return fields if len(fields) > 19 else None


def _start(pid: int) -> str | None:
    fields = _stat(pid)
    return fields[19] if fields else None


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def move_aside(path: Path, why: str) -> Path | None:
    """`path` renamed to `<path>.<why>-<n>`, the first `n` free; None when there was nothing."""
    if not path.exists():
        return None
    n = 1
    while (target := path.with_name(f"{path.name}.{why}-{n}")).exists():
        n += 1
    path.rename(target)
    return target


class Ledger:
    """The process groups started in one directory that are still running, on disk."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / LEDGER_FILE

    def read(self) -> list[dict[str, Any]]:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [g for g in doc if isinstance(g, dict)] if isinstance(doc, list) else []

    def started(self, pid: int) -> None:
        """`pid`, which leads a session and process group of its own, is running."""
        entry = {"pgid": int(pid), "start": _start(pid), "boot": boot_id()}
        try:
            atomic_write_json(self.path, [*self.read(), entry])
        except OSError as exc:
            log.warning("could not record process group %d in %s: %s", pid, self.path, exc)

    def ended(self, pid: int) -> None:
        """`pid`'s group has been killed and waited for."""
        left = [g for g in self.read() if g.get("pgid") != int(pid)]
        try:
            if left:
                atomic_write_json(self.path, left)
            else:
                self.path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not update %s: %s", self.path, exc)


def _members(pgid: int) -> list[int]:
    """The live processes of process group `pgid` in session `pgid`."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fields = _stat(entry.name)
        if fields and fields[0] != "Z" and fields[2] == str(pgid) and fields[3] == str(pgid):
            found.append(int(entry.name))
    return found


def _still_recorded(group: dict[str, Any]) -> bool:
    """Whether the group a ledger names is still that group, and still has a live member."""
    try:
        pgid = int(group["pgid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pgid <= 1 or not group.get("boot") or group.get("boot") != boot_id():
        return False
    leader = _stat(pgid)
    if leader is not None and leader[19] != group.get("start"):
        return False  # the pid belongs to a later process now
    return bool(_members(pgid))


def reap_ledger(directory: Path) -> list[int]:
    """Kill every process group the ledger in `directory` names that is still the one recorded,
    and remove the ledger; the groups killed."""
    ledger = Ledger(directory)
    killed = []
    for group in ledger.read():
        if not _still_recorded(group):
            continue
        pgid = int(group["pgid"])
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError as exc:
            log.warning("could not kill process group %d: %s", pgid, exc)
            continue
        killed.append(pgid)
        deadline = time.monotonic() + KILL_WAIT_S
        while _members(pgid) and time.monotonic() < deadline:
            time.sleep(POLL_S)
    ledger.path.unlink(missing_ok=True)
    if killed:
        log.warning("killed process groups %s left running in %s", killed, directory)
    return killed


def reap_run_root(run_root: Path, *, decided: str = "outcome.json") -> list[int]:
    """Every ledger under the duels of `run_root` that were not decided, reaped."""
    killed: list[int] = []
    if not Path(run_root).is_dir():
        return killed
    for duel_dir in sorted(Path(run_root).glob("*/*")):
        if not duel_dir.is_dir() or (duel_dir / decided).exists():
            continue
        for path in sorted(duel_dir.rglob(LEDGER_FILE)):
            killed += reap_ledger(path.parent)
    return killed
