"""The challenger queues: one per track, one entry per submission key, rewritten atomically.

A queue is local state, not the record. What the dashboard shows is `snapshot()`, written to the
store as `tracks/{track}/queue.json` (schema 4 `QueueSnapshot`) - unsigned and rewritten every cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ids import SubmissionRef
from .store.records import now_iso
from .store.writer import atomic_write_json, read_json


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


@dataclass
class QueueState:
    entries: list[QueueEntry] = field(default_factory=list)
    in_progress: InProgress | None = None
    block: int = 0


class Queue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state = self._load()

    def _load(self) -> QueueState:
        doc = read_json(self.path) or {}
        # Fields a newer or older writer added are ignored rather than refused: a queue file is
        # local state and must survive an upgrade.
        entries = [
            QueueEntry(**{k: v for k, v in e.items() if k in QueueEntry.__dataclass_fields__})
            for e in doc.get("entries", [])
        ]
        ip = doc.get("in_progress")
        in_progress = InProgress(**ip) if ip else None
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
        """Queue a submission at the back. Re-adding the same `repo@revision` moves it to the back
        rather than queueing it twice."""
        ref = SubmissionRef.make(repo, revision)
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
        self.save()
        return entry, len(self.state.entries)

    def remove(self, key: str) -> bool:
        before = len(self.state.entries)
        self.state.entries = [e for e in self.state.entries if e.key != key]
        self.save()
        return len(self.state.entries) != before

    def peek(self) -> QueueEntry | None:
        return self.state.entries[0] if self.state.entries else None

    def pop(self) -> QueueEntry | None:
        if not self.state.entries:
            return None
        entry = self.state.entries.pop(0)
        self.save()
        return entry

    def start(self, event_id: str, challenger: SubmissionRef, *, now: str | None = None) -> None:
        self.state.in_progress = InProgress(
            event_id=event_id, challenger=challenger.as_dict(), started_at=now or now_iso()
        )
        self.save()

    def finish(self) -> None:
        self.state.in_progress = None
        self.save()

    def advance_block(self) -> int:
        self.state.block += 1
        self.save()
        return self.state.block

    @property
    def block(self) -> int:
        return self.state.block

    def entries(self) -> list[QueueEntry]:
        return list(self.state.entries)

    # ---------------------------------------------------------------- published view
    def snapshot(
        self,
        track: str,
        king: SubmissionRef | None,
        schema: int,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": schema,
            "track": track,
            "block": self.state.block,
            "written_at": now or now_iso(),
            "king": king.as_dict() if king else None,
            "in_progress": asdict(self.state.in_progress) if self.state.in_progress else None,
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
                for i, e in enumerate(self.state.entries)
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
