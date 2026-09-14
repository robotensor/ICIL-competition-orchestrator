"""The orchestrator loop: queue -> duel -> publish -> mirror, for every track. One per store.

Ported from icilval's `daemon.py`. Each step takes one track's next entry and runs it to the end:

- A duel in progress (the queue file's `in_progress`) is resumed first, from the request its run
  directory holds: a daemon that was killed picks up where it stopped, with every unit that had
  finished kept.
- An empty throne is taken by genesis: the track's baseline when `baselines` names one, otherwise
  the queue's first entry.
- The king queued again is dropped: a duel against itself decides nothing.
- Otherwise the entry duels the king, in the next block (one past both the queue's counter and
  the head's). The entry leaves the queue and the duel is marked in progress in one write
  (`Queue.take`), so a restart finds one or the other.

A duel that publishes, is refused or void, or fails for the harness finishes its entry: the queue
moves on (as in icilval; a failed entry is queued again by hand). An interrupt does not finish it,
which is what lets the next daemon resume it. Nor does a harness that is only unavailable for now
(`HarnessUnavailable`: the benchmark not installed or not the pinned one, Docker down, the Hub
unreachable): the duel stays in progress with everything it had done, and a later step resumes it;
a duel in progress is not resumed while its track's benchmark is not ready. A duel whose king lost the crown while it was stopped
(`CrownMoved`) is discarded - its run moved aside, nothing published - and its challenger's entry
goes back to the head of the queue, to duel the king who holds the crown now. A track whose
benchmark is not installed, or not the pinned one, is skipped with a warning and its queue left as
it is.

Tracks are served round-robin by one process: the store has one writer, held by `store_lock` for
the daemon's lifetime.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .duel.orchestrate import (
    REQUEST_FILE,
    CrownMoved,
    DuelFailed,
    DuelRequest,
    HarnessUnavailable,
    Orchestrator,
)
from .ids import SubmissionRef
from .queue import InProgress, Queue, QueueEntry, Queues
from .store.records import now_iso
from .store.writer import read_json, store_lock

log = logging.getLogger(__name__)

#: After a step crashes, the daemon waits this long, doubling with each crash in a row up to
#: `max_backoff_s`, so an error that does not go away is retried without spinning.
BACKOFF_START_S = 1.0
MAX_BACKOFF_S = 300.0


class Daemon:
    def __init__(
        self,
        orchestrator: Orchestrator,
        queues: Queues,
        *,
        idle_sleep_s: float = 15.0,
        max_backoff_s: float = MAX_BACKOFF_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.orchestrator = orchestrator
        self.spec = orchestrator.spec
        self.store = orchestrator.store
        self.queues = queues
        self.idle_sleep_s = idle_sleep_s
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep
        #: Tracks whose baseline genesis did not publish: not retried by this process.
        self.stalled: set[str] = set()
        #: Held over computing a queue snapshot and writing it, which the submission intake does
        #: from its own threads too: a snapshot computed before another thread's change and written
        #: after that thread's own would publish the queue without the change.
        self._publishing = threading.Lock()

    # -- state ------------------------------------------------------------------------------

    def current_king(self, track: str) -> SubmissionRef | None:
        head = self.store.head(track) or {}
        return SubmissionRef.from_dict(head.get("king"))

    def next_block(self, track: str, queue: Queue) -> int:
        head = self.store.head(track) or {}
        return max(queue.block, int(head.get("block") or 0)) + 1

    def publish_queue(self, track: str, *, mirror: bool = True) -> None:
        """`tracks/{track}/queue.json` as the queue is now, pushed to the mirror unless `mirror` is
        False: a caller that must not wait on a push (the intake, answering a request) leaves the
        file touched, and the daemon's next push carries it."""
        with self._publishing:
            snapshot = self.queues[track].snapshot(
                track, self.current_king(track), int(self.spec.store["schema"])
            )
            self.store.write_queue(track, snapshot)
        if mirror:
            self._mirror()

    def _mirror(self) -> None:
        self.orchestrator.push_touched()

    # -- one step ---------------------------------------------------------------------------

    def step(self, track: str) -> bool:
        """Run one duel of `track`, resumed or taken from its queue. False when there was none."""
        queue = self.queues[track]
        in_progress = queue.reload().in_progress
        if in_progress is not None:
            if not self._benchmarks_ready(track):
                return False  # the duel stays in progress, for when its benchmark is back
            return self._resume(track, queue, in_progress)
        if track in self.stalled:
            return False
        king = self.current_king(track)
        baseline = self.spec.baseline(track)
        if king is None and baseline is not None:
            return self._baseline_genesis(track, queue, baseline)
        entry = queue.peek()
        if entry is None:
            return False
        if not self._benchmarks_ready(track):
            return False
        if king is not None and king.key == entry.key:
            log.info("%s: %s already holds the crown; dropping the entry", track, entry.ref.entry)
            queue.remove(entry.key)
            self.publish_queue(track)
            return True
        req = DuelRequest(
            track=track,
            challenger=entry.ref,
            king=king,
            size=entry.duel_size,
            block=self.next_block(track, queue),
        )
        self.orchestrator.record_request(req)
        if queue.take(entry.key, block=req.block, event_id=req.event_id(self.spec)) is None:
            return True  # removed from the queue meanwhile
        self.publish_queue(track)
        return self._run(track, queue, req)

    def step_all(self) -> bool:
        """One step for every track, in turn. False when no track had anything to do."""
        ran = False
        for track in self.queues.tracks:
            ran |= self.step(track)
        return ran

    def run(self, *, once: bool = False) -> None:
        """Hold the store and serve the queues until killed, or for one round with `once`."""
        with store_lock(self.store.root):
            key = self.store.signer.verify_key_hex if self.store.signer else "?"
            log.info("orchestrator %s serving %s", key[:12], self.store.root)
            self.orchestrator.reap_orphans()
            for track in self.queues.tracks:
                self.publish_queue(track)
            crashes = 0
            while True:
                try:
                    ran = self.step_all()
                except Exception:  # noqa: BLE001 - one bad step must not stop every track
                    crashes += 1
                    if once:
                        log.exception("a daemon step crashed")
                        return
                    delay = self.backoff_s(crashes)
                    log.exception(
                        "a daemon step crashed (%d in a row); retrying in %gs", crashes, delay
                    )
                    self._sleep(delay)
                    continue
                crashes = 0
                if once:
                    return
                if not ran:
                    self._sleep(self.idle_sleep_s)

    def backoff_s(self, crashes: int) -> float:
        """The wait after `crashes` steps in a row crashed: doubling from `BACKOFF_START_S`,
        never more than `max_backoff_s`."""
        return min(self.max_backoff_s, BACKOFF_START_S * 2.0 ** min(max(crashes, 1) - 1, 60))

    # -- helpers ----------------------------------------------------------------------------

    def _run(self, track: str, queue: Queue, req: DuelRequest) -> bool:
        """Run `req`, which is in progress, and settle its entry. False when the harness was not
        there to run it: the duel stays in progress, with everything it had done, to be retried."""
        try:
            result = self.orchestrator.run(req)
        except HarnessUnavailable as exc:
            log.warning(
                "%s: duel %s kept in progress for a later try: %s",
                track,
                req.event_id(self.spec)[:16],
                exc,
            )
            return False
        except CrownMoved as exc:
            entry = queue.put_back(self._entry_of(track, req))
            again = (
                f"{entry.ref.entry} is queued again at the head" if entry else "nothing requeued"
            )
            log.warning(
                "%s: duel %s discarded: %s (its run is at %s); %s",
                track,
                req.event_id(self.spec)[:16],
                exc.reason,
                exc.moved_to,
                again,
            )
            self.publish_queue(track)
            return True
        except DuelFailed as exc:
            log.error("%s: duel %s failed: %s", track, req.event_id(self.spec)[:16], exc)
        else:
            log.info("%s: %s %s: %s", track, req.kind, result.status, result.reason)
        # Not in a `finally`: an interrupt leaves the duel in progress, to be resumed.
        queue.finish()
        self.publish_queue(track)
        return True

    def _resume(self, track: str, queue: Queue, in_progress: InProgress) -> bool:
        path = Path(self.orchestrator.run_root) / track / in_progress.event_id[:16] / REQUEST_FILE
        doc = read_json(path)
        try:
            req = DuelRequest.from_dict(doc) if isinstance(doc, dict) else None
        except (KeyError, TypeError, ValueError):
            req = None
        if req is None or req.event_id(self.spec) != in_progress.event_id:
            log.error(
                "%s: duel %s is in progress but %s does not hold its request; finishing it",
                track,
                in_progress.event_id[:16],
                path,
            )
            queue.finish()
            self.publish_queue(track)
            return True
        log.info("%s: resuming duel %s", track, in_progress.event_id[:16])
        return self._run(track, queue, req)

    def _entry_of(self, track: str, req: DuelRequest) -> QueueEntry | None:
        """The entry a stale duel goes back as when its in-progress mark kept none: the request's
        challenger at the request's size. None for the track's baseline, which no entry asked for."""
        baseline = self.spec.baseline(track)
        if req.king is None and baseline is not None:
            if (baseline.get("repo"), baseline.get("revision")) == (
                req.challenger.repo,
                req.challenger.revision,
            ):
                return None
        return QueueEntry(
            key=req.challenger.key,
            repo=req.challenger.repo,
            revision=req.challenger.revision,
            commit_block=req.block,
            duel_size=req.size,
            accepted_at=now_iso(),
            source="requeued",
        )

    def _baseline_genesis(self, track: str, queue: Queue, baseline: dict[str, Any]) -> bool:
        try:
            ref = SubmissionRef.resolved(str(baseline.get("repo")), str(baseline.get("revision")))
        except ValueError as exc:
            log.error("%s: the baseline cannot take the throne: %s", track, exc)
            self.stalled.add(track)
            return False
        if not self._benchmarks_ready(track):
            return False
        req = DuelRequest(track, ref, None, None, block=self.next_block(track, queue))
        self.orchestrator.record_request(req)
        queue.set_block(req.block)
        queue.start(req.event_id(self.spec), ref)
        self.publish_queue(track)
        if not self._run(track, queue, req):
            return False  # in progress still, retried once the harness is back
        if self.current_king(track) is None:
            log.error("%s: the baseline's genesis did not publish; the track waits", track)
            self.stalled.add(track)
        return True

    def _benchmarks_ready(self, track: str) -> bool:
        from .benchmarks.plugins import BenchmarkRefused

        for name in self.spec.benchmarks_of(track):
            try:
                self.orchestrator.resolve(name)
            except BenchmarkRefused as exc:
                log.warning("%s: skipped, %s", track, exc)
                return False
        return True
