"""The challenger queues: one per track, one entry per submission key, rewritten atomically.

A queue is local state, not the record. What the dashboard shows is `snapshot()`, written to the
store as `tracks/{track}/queue.json` (schema 4 `QueueSnapshot`) - unsigned and rewritten every cycle.

The file is the state, not this object: a long-lived duel loop and a `queue add` on the command
line hold the same queue. Every mutation takes a lock on the file, reloads, changes and writes, so
one writer cannot save a list the other has already added to; every read reloads too. A queue file
that exists but cannot be read is refused rather than taken for an empty queue, which would drop
the waiting challengers and reset the block counter every event id is derived from.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ids import SubmissionRef
from .store.records import now_iso
from .store.writer import atomic_write_json


@dataclass
class QueueEntry:
    key: str
    repo: str
    revision: str
    commit_block: int
    duel_size: str | None
    accepted_at: str
    source: str = ""

    @property
    def ref(self) -> SubmissionRef:
        return SubmissionRef(key=self.key, repo=self.repo, revision=self.revision)


@dataclass
class InProgress:
    event_id: str
    challenger: dict[str, str]
    started_at: str
    #: The queue entry the duel was taken from, to put back if the duel goes stale. Local state:
    #: never in the published snapshot. None for a baseline's genesis, which no entry asked for.
    entry: dict[str, Any] | None = None


@dataclass
class QueueState:
    entries: list[QueueEntry] = field(default_factory=list)
    in_progress: InProgress | None = None
    block: int = 0


class Queue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        #: Held over every load into `state` and every change saved from it. Threads can share one
        #: Queue (the intake's handlers do): a reader replacing `state` between a writer's change
        #: and its save would have the writer save the file without that change.
        self._thread_lock = threading.RLock()
        self.state = self._load()

    @contextmanager
    def _locked(self):
        """Exclusive across processes and threads for one queue file, over load, change and save."""
        lock = self.path.with_name(self.path.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                self.state = self._load()
                yield
                self.save()
            finally:
                os.close(fd)

    def reload(self) -> QueueState:
        """The file as it is now. The state returned is never changed afterwards: a writer loads its
        own, so a caller can read it after another thread has moved on."""
        with self._thread_lock:
            self.state = self._load()
            return self.state

    def _load(self) -> QueueState:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            doc = {}
        except (OSError, ValueError, RecursionError) as exc:
            raise ValueError(f"{self.path} is not a readable queue file: {exc}") from exc
        if not isinstance(doc, dict):
            raise ValueError(f"{self.path} is not a readable queue file: not a JSON object")
        # Fields a newer or older writer added are ignored rather than refused: a queue file is
        # local state and must survive an upgrade.
        entries = [
            QueueEntry(**{k: v for k, v in e.items() if k in QueueEntry.__dataclass_fields__})
            for e in doc.get("entries", [])
        ]
        ip = doc.get("in_progress")
        in_progress = (
            InProgress(**{k: v for k, v in ip.items() if k in InProgress.__dataclass_fields__})
            if ip
            else None
        )
        return QueueState(entries=entries, in_progress=in_progress, block=int(doc.get("block", 0)))

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "entries": [asdict(e) for e in self.state.entries],
                "in_progress": asdict(self.state.in_progress) if self.state.in_progress else None,
                "block": self.state.block,
            },
        )

    # ---------------------------------------------------------------- mutations
    def add(
        self,
        repo: str,
        revision: str,
        *,
        duel_size: str | None = None,
        source: str = "",
        now: str | None = None,
    ) -> tuple[QueueEntry, int]:
        """Queue a submission at the back, at a repo id and a resolved commit sha. Re-adding the
        same `repo@revision` moves it to the back rather than queueing it twice."""
        ref = SubmissionRef.resolved(repo, revision)
        with self._locked():
            self.state.entries = [e for e in self.state.entries if e.key != ref.key]
            entry = QueueEntry(
                key=ref.key,
                repo=repo,
                revision=revision,
                commit_block=self.state.block,
                duel_size=duel_size,
                accepted_at=now or now_iso(),
                source=source,
            )
            self.state.entries.append(entry)
            position = len(self.state.entries)
        return entry, position

    def offer(
        self,
        repo: str,
        revision: str,
        *,
        duel_size: str | None = None,
        source: str = "",
        now: str | None = None,
    ) -> tuple[QueueEntry, int, bool]:
        """`add`, except that a submission already waiting keeps its entry and its place.

        For an intake that can be sent the same entry twice - a retried request, a second click -
        where moving it to the back would cost it the place it had. Returns the entry, its position
        and whether it was queued now (False: it was already waiting, and nothing changed).
        """
        ref = SubmissionRef.resolved(repo, revision)
        with self._locked():
            waiting = [i for i, e in enumerate(self.state.entries) if e.key == ref.key]
            if not waiting:
                self.state.entries.append(
                    QueueEntry(
                        key=ref.key,
                        repo=repo,
                        revision=revision,
                        commit_block=self.state.block,
                        duel_size=duel_size,
                        accepted_at=now or now_iso(),
                        source=source,
                    )
                )
            index = waiting[0] if waiting else len(self.state.entries) - 1
            entry = self.state.entries[index]
        return entry, index + 1, not waiting

    def remove(self, key: str) -> bool:
        with self._locked():
            before = len(self.state.entries)
            self.state.entries = [e for e in self.state.entries if e.key != key]
            removed = len(self.state.entries) != before
        return removed

    def peek(self) -> QueueEntry | None:
        entries = self.reload().entries
        return entries[0] if entries else None

    def pop(self) -> QueueEntry | None:
        with self._locked():
            entry = self.state.entries.pop(0) if self.state.entries else None
        return entry

    def take(
        self, key: str, *, block: int, event_id: str, now: str | None = None
    ) -> QueueEntry | None:
        """Take the entry `key` off the queue and mark its duel in progress, in one write.

        Popping and marking separately would leave a window in which a killed orchestrator had
        dropped the entry without a trace of the duel it was taken for; with both in one write, a
        restart finds either the entry still queued or its duel in progress, and resumes it. The
        block counter moves to `block`, never back. None when the entry has gone meanwhile.
        """
        with self._locked():
            entry = next((e for e in self.state.entries if e.key == key), None)
            if entry is not None:
                self.state.entries = [e for e in self.state.entries if e.key != key]
                self.state.block = max(self.state.block, int(block))
                self.state.in_progress = InProgress(
                    event_id=event_id,
                    challenger=entry.ref.as_dict(),
                    started_at=now or now_iso(),
                    entry=asdict(entry),
                )
        return entry

    def put_back(self, fallback: QueueEntry | None = None) -> QueueEntry | None:
        """The duel in progress back at the head of the queue as the entry it was taken from, and
        nothing in progress, in one write; the entry put back.

        `fallback` stands in for an in-progress duel that recorded no entry (a file written before
        entries were kept); with neither, nothing is queued - a baseline's genesis has no entry.
        An entry of the same key queued meanwhile gives way to the one put back.
        """
        with self._locked():
            ip = self.state.in_progress
            entry = None
            if ip is not None and ip.entry:
                fields = QueueEntry.__dataclass_fields__
                entry = QueueEntry(**{k: v for k, v in ip.entry.items() if k in fields})
            elif ip is not None:
                entry = fallback
            if entry is not None:
                self.state.entries = [entry] + [e for e in self.state.entries if e.key != entry.key]
            self.state.in_progress = None
        return entry

    def start(self, event_id: str, challenger: SubmissionRef, *, now: str | None = None) -> None:
        with self._locked():
            self.state.in_progress = InProgress(
                event_id=event_id, challenger=challenger.as_dict(), started_at=now or now_iso()
            )

    def finish(self) -> None:
        with self._locked():
            self.state.in_progress = None

    def advance_block(self) -> int:
        with self._locked():
            self.state.block += 1
            block = self.state.block
        return block

    def claim_block(self, floor: int = 0) -> int:
        """The next block - one past both this counter and `floor`, the head's - for a duel no entry
        of this queue asked for (one run on the command line). The counter moves to it, so no duel
        the daemon runs later is numbered the same, and none shares its run directory."""
        with self._locked():
            self.state.block = max(self.state.block, int(floor)) + 1
            block = self.state.block
        return block

    def set_block(self, block: int) -> int:
        """For a rebuilt or seeded queue: the block a new entry is stamped with."""
        with self._locked():
            self.state.block = block
        return block

    @property
    def block(self) -> int:
        return self.reload().block

    def entries(self) -> list[QueueEntry]:
        return list(self.reload().entries)

    # ---------------------------------------------------------------- published view
    def snapshot(
        self,
        track: str,
        king: SubmissionRef | None,
        schema: int,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        state = self.reload()
        return {
            "schema": schema,
            "track": track,
            "block": state.block,
            "written_at": now or now_iso(),
            "king": king.as_dict() if king else None,
            "in_progress": (
                {
                    k: v
                    for k, v in asdict(state.in_progress).items()
                    if k in ("event_id", "challenger", "started_at")
                }
                if state.in_progress
                else None
            ),
            "entries": [
                {
                    "position": i + 1,
                    "key": e.key,
                    "repo": e.repo,
                    "revision": e.revision,
                    "commit_block": e.commit_block,
                    "duel_size": e.duel_size,
                    # Schema 4 requires it and the dashboard reads it. There is no model config
                    # check any more, so nothing can skip one.
                    "skip_model_config_check": False,
                    "accepted_at": e.accepted_at,
                }
                for i, e in enumerate(state.entries)
            ],
        }


class Queues:
    """One queue per track, as one file each in a directory.

    A track's queue is its own file because its lineage is its own: the block counter advances with
    that track's events, and a busy track must not hold up another's entries.
    """

    def __init__(self, root: str | Path, tracks: Sequence[str]):
        self.root = Path(root)
        if self.root.is_file():
            raise ValueError(
                f"{self.root} is a file: the queue is a directory with one file per track"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self._queues = {t: Queue(self.root / f"{t}.json") for t in tracks}

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(self._queues)

    def __getitem__(self, track: str) -> Queue:
        try:
            return self._queues[track]
        except KeyError:
            raise KeyError(
                f"unknown track {track!r}; the tracks are {', '.join(self._queues)}"
            ) from None

    def __contains__(self, track: object) -> bool:
        return track in self._queues

    def items(self):
        return self._queues.items()
