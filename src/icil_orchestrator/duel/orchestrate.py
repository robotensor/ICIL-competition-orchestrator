"""A duel from request to published record.

    fetching -> checking -> materializing -> evaluating(challenger) -> evaluating(king)
             -> publishing -> done | failed

Ported from icilval's `duel/orchestrate.py`, with a submission's policy reached through a
`PolicyRuntime` rather than loaded in-process, and prompts always materialized.

**Identity.** A duel is `(spec, track, challenger, king, size)`: its id is
`ids.duel_id(spec_version, track, challenger, king)` and its units are
`plugin_units(spec, track, duel_id, size)`, both pure functions of those, so anyone holding the
record can derive them again. Its event id adds the queue's block. Everything the duel keeps on
this host lives in `<run_root>/<track>/<event_id[:16]>/`:

    request.json                  what was asked, and when it started
    prompts/<unit_id>/prompt.npz  every unit's prompt, and prompts/manifest.jsonl
    <side>/results.jsonl          each side's finished units, and <side>/<unit_id>/ its run logs
    forfeit.json                  why the king forfeits, when it does
    outcome.json                  what was decided: published (with the record), refused or void
    failed.txt                    why the last attempt stopped, for a duel that did not finish

**Resuming.** Every stage is resumable from that directory: prompts are re-used after their hash
is checked, a side runs only the units its results file does not hold, and a duel first looks for
its own event id in the index, so a duel killed at any point and run again finishes with each unit
run once per side and at most one record. One found there is not run again: its head is rebuilt
from the index (a kill between the index line and the head's rewrite leaves the old king in it)
and its files are pushed to the mirror again. Once `outcome.json` exists, running the duel again
returns it.

**What stops a duel, and what that means.**

- `DuelFailed`: the harness could not run it (no benchmark, no Docker, the Hub unreachable, a
  bug). Nothing is published and nothing is decided; running it again resumes it. Its subclass
  `HarnessUnavailable` is the part of that which may pass by itself - a benchmark not installed or
  not the pinned one, Docker down, the Hub unreachable - which the daemon retries later.
- Refused: the challenger's submission cannot run (a bad manifest, an image that does not build,
  no `hello`). Nothing else runs - the king is not fetched and no prompt is made - nothing is
  published, and `outcome.json` says why.
- Void: the duel ran and cannot stand. The crown does not move, nothing is published, and
  `outcome.json` says why. A duel is void when more than `max_void_fraction` of its units are void
  (checked after materializing and after each side, so the king is not run for a duel the
  challenger's side already voided). A unit is void only for a harness cause (`side`); what a
  side's own submission does to a unit is that side's failure.
- A king whose submission is refused (its repository gone or private, an image that no longer
  builds, a manifest no longer valid) forfeits: every one of its units is a failure, the duel is
  scored and published as any other with the note `king forfeit: <reason>`, and the challenger
  takes the crown if its own average clears the margin. The forfeit is recorded in
  `forfeit.json`, so a resumed duel keeps it.
- `CrownMoved`: the track's head no longer names the king the duel was asked against (someone
  else was crowned while this duel was stopped). Checked before the duel starts or resumes, and
  again just before it publishes: its run directory is moved aside as `<dir>.stale-<n>`, nothing
  is published, and the caller queues the challenger again. A record against a king who has lost
  the crown would hand it back to him.
- A published duel moves the crown iff `score.crown_moves`.

**Genesis.** A track with no king (`baselines` null and nothing crowned yet) crowns its first
challenger by an event of kind `genesis`: its side is materialized and evaluated like any other,
and published with its own scores in the record's `king` slot, which is where a genesis names what
it crowns. A genesis is void on the same rule as a duel.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from ..benchmarks.units import plugin_units
from ..ids import SubmissionRef, duel_id, event_id
from ..live import LiveReporter, build_frame
from ..spec import Spec
from ..store.records import (
    duel_event,
    index_record,
    media_shas,
    now_iso,
    unit_tally,
    unit_verdict_from_unit,
)
from ..store.writer import Store, atomic_write_json, read_json
from . import score
from .materialize import Materialized, Prompt, materialize_units
from .orphans import move_aside, reap_run_root
from .runtime import PolicyRuntime, PreparedSubmission, RuntimeUnavailable, SubmissionRefused
from .side import read_results, run_side

log = logging.getLogger(__name__)

REQUEST_FILE = "request.json"
OUTCOME_FILE = "outcome.json"
FAILED_FILE = "failed.txt"
#: The king's refusal, recorded once: a resumed duel keeps the forfeit rather than asking again.
FORFEIT_FILE = "forfeit.json"
SIDES = ("challenger", "king")


class DuelFailed(RuntimeError):
    """The harness could not run the duel. Nothing was published or decided."""


class HarnessUnavailable(DuelFailed):
    """A part of the harness the duel needs is not there for now - its benchmark not installed or
    not the pinned one, Docker down, the Hub unreachable. Worth trying the same duel again later."""


class CrownMoved(RuntimeError):
    """The track's crown is no longer held by the king the duel was asked against. Nothing was
    published, the duel's run was moved aside (`moved_to`), and its challenger is to be queued
    again: a record against a king who no longer holds the crown would hand it back to him."""

    def __init__(self, req: DuelRequest, reason: str, moved_to: Path | None) -> None:
        self.req = req
        self.reason = reason
        self.moved_to = moved_to
        super().__init__(reason)


@dataclass(frozen=True)
class DuelRequest:
    track: str
    challenger: SubmissionRef
    #: None where the track has no king: the duel is then a genesis.
    king: SubmissionRef | None
    size: str | None
    #: The queue's block this duel is fought in; part of its event id.
    block: int

    @property
    def kind(self) -> str:
        return "duel" if self.king is not None else "genesis"

    def duel_id(self, spec: Spec) -> str:
        return duel_id(spec.version, self.track, self.challenger, self.king)

    def event_id(self, spec: Spec) -> str:
        return event_id(self.kind, self.track, self.block, self.duel_id(spec))

    def as_dict(self, spec: Spec) -> dict[str, Any]:
        return {
            "track": self.track,
            "kind": self.kind,
            "challenger": self.challenger.as_dict(),
            "king": self.king.as_dict() if self.king else None,
            "size": spec.size_of(self.track, self.size),
            "block": self.block,
            "duel_id": self.duel_id(spec),
            "event_id": self.event_id(spec),
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> DuelRequest:
        challenger = SubmissionRef.from_dict(doc["challenger"])
        if challenger is None:
            raise ValueError("a duel request names no challenger")
        return cls(
            track=str(doc["track"]),
            challenger=challenger,
            king=SubmissionRef.from_dict(doc.get("king")),
            size=doc.get("size"),
            block=int(doc["block"]),
        )


@dataclass
class DuelResult:
    #: published or void.
    status: str
    kind: str
    event_id: str
    duel_id: str
    reason: str
    run_dir: Path
    record: dict[str, Any] | None = None
    units: list[dict[str, Any]] = field(default_factory=list)

    @property
    def published(self) -> bool:
        return self.status == "published"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "event_id": self.event_id,
            "duel_id": self.duel_id,
            "reason": self.reason,
            "run_dir": str(self.run_dir),
            "record": self.record,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> DuelResult:
        return cls(
            status=str(doc["status"]),
            kind=str(doc["kind"]),
            event_id=str(doc["event_id"]),
            duel_id=str(doc["duel_id"]),
            reason=str(doc["reason"]),
            run_dir=Path(doc["run_dir"]),
            record=doc.get("record"),
            units=list(doc.get("units") or []),
        )


@dataclass
class _Duel:
    """One run of one duel: what the phases share, and what a live frame shows."""

    req: DuelRequest
    duel_id: str
    event_id: str
    size: str
    run_dir: Path
    started_at: str
    deadline: float
    units: list[dict[str, Any]] = field(default_factory=list)
    unit_defs: list[dict[str, Any]] = field(default_factory=list)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    prepared: dict[str, PreparedSubmission] = field(default_factory=dict)
    prompts: Materialized | None = None
    sides: dict[str, dict[str, Any]] = field(default_factory=dict)
    phase: str = "fetching"
    side: str | None = None
    current: dict[str, Any] | None = None
    recent_media: dict[str, Any] | None = None
    message: str = ""
    #: Finished clips not yet in the store: `(side, unit_id) -> path`.
    pending_media: dict[tuple[str, str], Path] = field(default_factory=dict)
    #: Why the king forfeits, when its submission was refused.
    forfeit: str | None = None

    def row(self, unit_id: str) -> dict[str, Any]:
        return next(u for u in self.units if u["unit_id"] == unit_id)


class Orchestrator:
    """Runs duels into one store. The caller holds the store's lock (`store_lock`)."""

    def __init__(
        self,
        spec: Spec,
        store: Store,
        runtime: PolicyRuntime,
        run_root: str | Path,
        *,
        live: LiveReporter | None = None,
        mirror: Any | None = None,
        resolve: Callable[[str], Any] | None = None,
    ) -> None:
        if store.signer is None:
            raise ValueError("the store has no signer; a duel cannot publish without one")
        self.spec = spec
        self.store = store
        self.runtime = runtime
        self.run_root = Path(run_root)
        runtime.bind(store=store.root, runs=self.run_root)
        self.live = live or LiveReporter(spec, None, None)
        self.mirror = mirror
        if resolve is None:
            from ..benchmarks.plugins import load

            def resolve(name: str) -> Any:
                return load(spec, name)

        self.resolve = resolve

    # -- the run directory ------------------------------------------------------------------

    def reap_orphans(self) -> list[str]:
        """End what an orchestrator of this store and run root, killed outright, left running:
        its policy containers, and the process groups its undecided duels' ledgers name. Call it
        holding the store's lock, before any duel runs; what was reaped, by name."""
        reaped = [f"container {name}" for name in self.runtime.reap()]
        reaped += [f"process group {pgid}" for pgid in reap_run_root(self.run_root)]
        if reaped:
            log.warning("reaped what a killed orchestrator left running: %s", ", ".join(reaped))
        return reaped

    def run_dir(self, req: DuelRequest) -> Path:
        return self.run_root / req.track / req.event_id(self.spec)[:16]

    def record_request(self, req: DuelRequest) -> dict[str, Any]:
        """Write `request.json` for `req` unless it is there. A run directory started for another
        duel of the same id (the same pair and block at another size) is moved aside as
        `<dir>.stale-<n>` and logged, rather than refused on every attempt."""
        if req.track not in self.spec.tracks:
            raise DuelFailed(f"unknown track {req.track!r}")
        path = self.run_dir(req) / REQUEST_FILE
        wanted = req.as_dict(self.spec)
        existing = read_json(path)
        if isinstance(existing, dict):
            if self.holds_request(req):
                return existing
            # The same pair at the same block, asked for differently (another size): whatever
            # that run was, it is not this duel. Kept aside for its logs, never resumed as this.
            moved = move_aside(self.run_dir(req), "stale")
            log.warning(
                "%s held another duel's request (size %s, not %s); moved it aside to %s",
                path,
                existing.get("size"),
                wanted["size"],
                moved,
            )
        doc = {**wanted, "started_at": now_iso()}
        atomic_write_json(path, doc)
        return doc

    def holds_request(self, req: DuelRequest) -> bool:
        """Whether `req`'s run directory was started for exactly this duel."""
        doc = read_json(self.run_dir(req) / REQUEST_FILE)
        if not isinstance(doc, dict):
            return False
        return {k: v for k, v in doc.items() if k != "started_at"} == req.as_dict(self.spec)

    def outcome(self, req: DuelRequest) -> DuelResult | None:
        doc = read_json(self.run_dir(req) / OUTCOME_FILE)
        return DuelResult.from_dict(doc) if isinstance(doc, dict) else None

    def _check_crown(self, req: DuelRequest, duel: _Duel | None = None) -> None:
        """`CrownMoved`, with the duel's run moved aside, unless the track's head still names the
        king `req` was asked against (no king at all, for a genesis)."""
        head = self.store.head(req.track) or {}
        holder = head.get("king") or None
        wanted = req.king.as_dict() if req.king is not None else None
        if holder == wanted:
            return
        run_dir = self.run_dir(req)
        moved = move_aside(run_dir, "stale")
        who = holder.get("repo") if isinstance(holder, dict) else "nobody"
        was = req.king.entry if req.king is not None else "nobody"
        reason = f"the crown moved from {was} to {who} since this duel was asked for"
        log.warning("duel %s is stale: %s; its run is at %s", run_dir.name, reason, moved)
        if duel is not None:
            self._post(duel, force=True, phase="failed", side=None, message=f"stale: {reason}")
        raise CrownMoved(req, reason, moved)

    # -- the duel ---------------------------------------------------------------------------

    def run(self, req: DuelRequest) -> DuelResult:
        """Run `req` to a published record or a void outcome, resuming what an earlier run left.

        `DuelFailed` when the harness cannot; an interrupt (`KeyboardInterrupt`, a kill) passes
        through untouched, and running the duel again resumes it.
        """
        request = self.record_request(req)
        decided = self.outcome(req)
        if decided is not None:
            log.info("duel %s was already decided: %s", decided.event_id[:16], decided.status)
            return decided
        existing = self._published(req.track, req.event_id(self.spec))
        if existing is not None:
            return self._republished(req, existing)
        self._check_crown(req)
        run_dir = self.run_dir(req)
        # The duel's budget is spent by work, not by the time a stopped orchestrator was down.
        spent = _work_seconds(run_dir)
        duel = _Duel(
            req=req,
            duel_id=req.duel_id(self.spec),
            event_id=req.event_id(self.spec),
            size=self.spec.size_of(req.track, req.size),
            run_dir=run_dir,
            started_at=str(request["started_at"]),
            deadline=time.monotonic() + float(self.spec.budgets["duel_wall_seconds"]) - spent,
        )
        try:
            result = self._run(duel)
        except CrownMoved:
            raise
        except DuelFailed as exc:
            self._failed(duel, str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - reported, then raised as the harness's failure
            log.exception("duel %s crashed", duel.event_id[:16])
            self._failed(duel, f"{type(exc).__name__}: {exc}")
            raise DuelFailed(f"{type(exc).__name__}: {exc}") from exc
        (run_dir / FAILED_FILE).unlink(missing_ok=True)
        return result

    def _run(self, duel: _Duel) -> DuelResult:
        spec, req = self.spec, duel.req
        self._derive(duel)
        sides = self._sides(req)

        # ---- fetching and checking: a refused challenger decides the duel before anything runs;
        # a refused king forfeits, once, and a resumed duel keeps the forfeit it recorded
        refused: dict[str, str] = {}
        forfeit = read_json(duel.run_dir / FORFEIT_FILE)
        if req.king is not None and isinstance(forfeit, dict) and forfeit.get("reason"):
            refused["king"] = str(forfeit["reason"])
        self._post(duel, force=True, phase="fetching", message="fetching the submissions")
        fetched: dict[str, Any] = {}
        for side, ref in sides:
            if side in refused:
                continue
            try:
                fetched[side] = self.runtime.fetch(ref, workdir=duel.run_dir / side)
            except SubmissionRefused as exc:
                refused[side] = str(exc)
                if side == "challenger":
                    break
            except RuntimeUnavailable as exc:
                raise HarnessUnavailable(f"fetching the {side}: {exc}") from exc
        self._post(duel, force=True, phase="checking", message="checking and building")
        for side, _ in sides:
            if "challenger" in refused:
                break
            if side in refused or side not in fetched:
                continue
            self._post(duel, message=f"checking the {side}")
            try:
                duel.prepared[side] = self.runtime.prepare(
                    fetched[side], workdir=duel.run_dir / side / "check"
                )
            except SubmissionRefused as exc:
                refused[side] = str(exc)
            except RuntimeUnavailable as exc:
                raise HarnessUnavailable(f"checking the {side}: {exc}") from exc
        for side, ref in sides:
            duel.sides[side] = self._side_meta(duel, side, ref, refused.get(side))
        if "challenger" in refused:
            return self._refused(
                duel, f"the challenger's submission was refused: {refused['challenger']}"
            )
        duel.forfeit = refused.get("king")
        if duel.forfeit is not None:
            log.warning("duel %s: the king forfeits: %s", duel.event_id[:16], duel.forfeit)
            atomic_write_json(duel.run_dir / FORFEIT_FILE, {"side": "king", "reason": duel.forfeit})

        # ---- materializing
        self._post(
            duel, force=True, phase="materializing", message="materializing the demonstrations"
        )
        duel.prompts = materialize_units(
            spec,
            duel.unit_defs,
            duel.run_dir / "prompts",
            benchmark_of=lambda unit: duel.benchmarks[unit["benchmark"]],
            deadline=duel.deadline,
            on_prompt=lambda unit, prompt: self._on_prompt(duel, unit, prompt),
        )
        self.push_touched()
        if score.too_void(duel.units, spec.max_void_fraction(req.track)):
            return self._void(duel, f"{duel.prompts.void} of {len(duel.units)} prompts are void")

        # ---- evaluating
        for side, _ in sides:
            self._evaluate(duel, side)
            if score.too_void(duel.units, spec.max_void_fraction(req.track)):
                void = sum(1 for u in duel.units if u["void"])
                return self._void(
                    duel, f"{void} of {len(duel.units)} units are void after the {side}'s side"
                )
        return self._publish(duel)

    # -- phases -----------------------------------------------------------------------------

    def _derive(self, duel: _Duel) -> None:
        """The units: a benchmark that is not installed stops the duel before anything is fetched."""
        from ..benchmarks.plugins import BenchmarkRefused
        from ..benchmarks.units import DerivationError

        req = duel.req
        try:
            for name in self.spec.benchmarks_of(req.track):
                duel.benchmarks[name] = self.resolve(name)
            duel.unit_defs = plugin_units(
                self.spec, req.track, duel.duel_id, duel.size, resolve=duel.benchmarks.__getitem__
            )
        except BenchmarkRefused as exc:
            raise HarnessUnavailable(str(exc)) from exc
        except DerivationError as exc:
            raise DuelFailed(str(exc)) from exc
        view = self.spec.demo_view(req.track)
        duel.units = []
        for unit in duel.unit_defs:
            row = unit_verdict_from_unit(unit, view)
            row["prompt_sha256"] = None
            duel.units.append(row)

    def _sides(self, req: DuelRequest) -> list[tuple[str, SubmissionRef]]:
        sides = [("challenger", req.challenger)]
        if req.king is not None:
            sides.append(("king", req.king))
        return sides

    def _on_prompt(self, duel: _Duel, unit: Mapping[str, Any], prompt: Prompt) -> None:
        row = duel.row(prompt.unit_id)
        if prompt.void:
            row["void"] = True
            row["king_error"] = row["challenger_error"] = prompt.error
        else:
            row["prompt"]["sha256"] = row["prompt_sha256"] = prompt.sha256
            if prompt.demo_clip is not None and self.spec.media.get("demo_video", True):
                row["demo_video"] = self.store.put_media(prompt.demo_clip, self.spec.video_format)
        done = sum(1 for u in duel.units if u["prompt_sha256"] or u["void"])
        self._post(duel, message=f"materialized {done} of {len(duel.units)}: {prompt.unit_id}")

    def _evaluate(self, duel: _Duel, side: str) -> None:
        side_dir = duel.run_dir / side
        self._post(duel, force=True, phase="evaluating", side=side, message=f"evaluating {side}")
        # What an earlier run of this duel finished is merged before anything new runs.
        for record in read_results(side_dir).values():
            self._merge(duel, side, record)
        other = _other(side)
        void_units = {
            unit_id: f"void on the {other}'s side: {record.get('error')}"
            for unit_id, record in read_results(duel.run_dir / other).items()
            if record.get("void")
        }
        run_side(
            self.spec,
            side=side,
            units=duel.unit_defs,
            prompts=duel.prompts or Materialized(root=duel.run_dir / "prompts", prompts={}),
            side_dir=side_dir,
            benchmark_of=lambda unit: duel.benchmarks[unit["benchmark"]],
            runtime=self.runtime,
            prepared=duel.prepared.get(side),
            refused=duel.forfeit if side == "king" else None,
            deadline=duel.deadline,
            budget_s=side_share(
                self.spec.budgets,
                sum(p.wall_s for p in duel.prompts.prompts.values()) if duel.prompts else 0.0,
                len(self._sides(duel.req)),
            ),
            void_units=void_units,
            on_start=lambda unit: self._on_start(duel, side, unit),
            on_unit=lambda unit, record: self._on_unit(duel, side, record),
        )
        self._flush_media(duel, force=True)
        results = read_results(side_dir)
        meta = duel.sides.setdefault(side, {})
        meta["wall_seconds"] = round(sum(float(r.get("wall_s") or 0) for r in results.values()), 3)
        meta["units"] = len(results)
        meta["void"] = sum(1 for r in results.values() if r.get("void"))

    def _on_start(self, duel: _Duel, side: str, unit: Mapping[str, Any]) -> None:
        duel.current = {
            "unit_id": unit["unit_id"],
            "skill": unit["skill"],
            "task": unit["task"],
            "task_label": unit.get("task_label"),
            "instance": unit["instance"],
            "side": side,
            "demo_video": duel.row(str(unit["unit_id"])).get("demo_video"),
        }
        self._post(duel, message=f"{side}: {unit['unit_id']}")

    def _on_unit(self, duel: _Duel, side: str, record: dict[str, Any]) -> None:
        self._merge(duel, side, record)
        self._flush_media(duel)
        state = "void" if record["void"] else ("ok" if record["success"] else "fail")
        duel.current = None
        self._post(duel, message=f"{side}: {record['unit_id']} {state}")

    def _merge(self, duel: _Duel, side: str, record: Mapping[str, Any]) -> None:
        """One side's unit record into the published row. Void on either side is void for both."""
        row = duel.row(str(record["unit_id"]))
        row[f"{side}_error"] = record.get("error")
        if record.get("void"):
            row["void"] = True
            row[f"{side}_success"] = None
        else:
            row[f"{side}_success"] = bool(record.get("success"))
            row[f"{side}_steps"] = record.get("steps")
            row[f"{side}_progress"] = _rate(record.get("progress"))
            row[f"{side}_metric"] = record.get("metric")
            clip = record.get("clip")
            if clip and not row.get(f"{side}_video"):
                duel.pending_media[(side, row["unit_id"])] = duel.run_dir / side / str(clip)
        if row["void"]:
            # Neither side is scored on it, and the record does not show one as if it were; what
            # the other side did stays in its results file and its error.
            row["king_success"] = row["challenger_success"] = None
            row["outcome"] = "tie"
        else:
            row["outcome"] = score.paired_outcome(
                row.get("king_success"), row.get("challenger_success")
            )

    def _flush_media(self, duel: _Duel, *, force: bool = False) -> None:
        """Finished clips into the store every `live.media_flush_units` units, so the dashboard can
        play them while the duel runs; mirrored when there is a mirror."""
        every = max(1, int(self.spec.live["media_flush_units"]))
        if not duel.pending_media or (not force and len(duel.pending_media) < every):
            return
        for (side, unit_id), path in list(duel.pending_media.items()):
            del duel.pending_media[(side, unit_id)]
            if not path.is_file():
                continue
            sha = self.store.put_media(path, self.spec.video_format)
            row = duel.row(unit_id)
            row[f"{side}_video"] = sha
            duel.recent_media = {
                "unit": {
                    "unit_id": unit_id,
                    "skill": row["skill"],
                    "task": row["task"],
                    "task_label": row.get("task_label"),
                    "instance": row["instance"],
                },
                "side": side,
                "demo_video": row.get("demo_video"),
                "video": sha,
                "success": row.get(f"{side}_success"),
            }
        self.push_touched()

    def _side_meta(
        self, duel: _Duel, side: str, ref: SubmissionRef, refused: str | None
    ) -> dict[str, Any]:
        prepared = duel.prepared.get(side)
        return {
            **ref.as_dict(),
            **(prepared.as_side() if prepared else {"commit": None, "base_image_digest": None}),
            "runtime": self.runtime.name,
            "refused": refused,
        }

    def _publish(self, duel: _Duel) -> DuelResult:
        spec, req = self.spec, duel.req
        track, skills, margin = req.track, spec.skills(req.track), spec.score_margin(req.track)
        self._post(duel, force=True, phase="publishing", side=None, message="publishing the record")
        existing = self._published(track, duel.event_id)
        if existing is not None:
            return self._republished(req, existing)
        self._check_crown(req, duel)
        if req.kind == "genesis":
            units = [_as_king(u) for u in duel.units]
            reason = "genesis"
            king_scores = score.skill_scores(units, "king", skills)
            challenger_scores, dethroned = None, False
            king, challenger, new_king = req.challenger, None, None
            sides = {"king": duel.sides.get("challenger", {})}
            scoring = {"reason": reason, "score_margin": margin, "delta_points": None}
        else:
            units = duel.units
            verdict = score.verdict(units, margin, skills)
            reason = verdict.reason
            king_scores, challenger_scores = verdict.king_scores, verdict.challenger_scores
            dethroned = verdict.dethroned
            king, challenger = req.king, req.challenger
            new_king = req.challenger if dethroned else None
            sides = duel.sides
            scoring = verdict.as_dict()
        record = index_record(
            schema=int(spec.store["schema"]),
            event_id=duel.event_id,
            kind=req.kind,
            track=track,
            block=req.block,
            finished_at=now_iso(),
            king=king,
            challenger=challenger,
            king_scores=king_scores,
            challenger_scores=challenger_scores,
            score_margin=margin,
            dethroned=dethroned,
            new_king=new_king,
            tally=unit_tally(units),
            media_count=len(media_shas(units)),
            duel_size=duel.size,
            duel_id=duel.duel_id,
        )
        event = duel_event(
            record,
            spec_version=spec.version,
            spec_fingerprint=spec.fingerprint,
            units=units,
            units_per_skill=spec.units_per_skill(track, duel.size),
            started_at=duel.started_at,
            wall_seconds=_seconds_since(duel.started_at),
            sides=sides,
            demonstration=spec.demonstration(track),
            prompts=self._prompts_published(duel),
            notes=[_note(req.kind, scoring)]
            + ([f"king forfeit: {duel.forfeit}"] if duel.forfeit else []),
        )
        event["runtime"] = self.runtime.name
        event["scoring"] = {
            **scoring,
            "void_fraction": score.void_fraction(units),
            "max_void_fraction": spec.max_void_fraction(track),
        }
        event["benchmarks"] = self._benchmark_info(duel)
        self.store.write_event(track, event)
        record["seq"] = self.store.append(track, record)
        self.push_touched()
        result = DuelResult(
            status="published",
            kind=req.kind,
            event_id=duel.event_id,
            duel_id=duel.duel_id,
            reason=reason,
            run_dir=duel.run_dir,
            record=record,
            units=units,
        )
        atomic_write_json(duel.run_dir / OUTCOME_FILE, result.as_dict())
        moved = "moves" if record.get("dethroned") or req.kind == "genesis" else "stays"
        self._post(duel, force=True, phase="done", message=f"the crown {moved}: {reason}")
        return result

    def _prompts_published(self, duel: _Duel) -> list[dict[str, Any]]:
        """The event's `prompts`: every prompt both sides ran from, by sha256, with the unit's
        demonstration clip only where the store holds it (`media.demo_video`) - a signed event
        never names a clip the store does not have."""
        if duel.prompts is None:
            return []
        return [
            {**entry, "demo_video": duel.row(str(entry["unit_id"])).get("demo_video")}
            for entry in duel.prompts.manifest()
        ]

    def _republished(self, req: DuelRequest, record: dict[str, Any]) -> DuelResult:
        """A duel an earlier run published before it stopped: nothing is fetched, run or published
        again. Its head is rebuilt from the index - a kill between the index line and the head's
        rewrite leaves the head naming the king before it, whom the next duel would face - and its
        files are pushed to the mirror again, since a kill before the push leaves them off it."""
        track, eid = req.track, str(record["event_id"])
        log.info(
            "duel %s is published as seq %s already; healing its head", eid[:16], record["seq"]
        )
        self.store.rebuild_head(track)
        event = self.store.event(track, eid) or {}
        units = [u for u in event.get("units") or [] if isinstance(u, dict)]
        paths = [
            self.store.event_path(track, eid),
            self.store.index_part_path(track, self.store.part_of(int(record["seq"]))),
            self.store.head_path(track),
            *(self.store.media_path(sha, self.spec.video_format) for sha in media_shas(units)),
        ]
        for path in paths:
            if path.is_file():
                self.store.touch(path)
        self.push_touched()
        scoring = event.get("scoring") if isinstance(event.get("scoring"), dict) else {}
        result = DuelResult(
            status="published",
            kind=str(record.get("kind") or req.kind),
            event_id=eid,
            duel_id=req.duel_id(self.spec),
            reason=str(scoring.get("reason") or record.get("kind")),
            run_dir=self.run_dir(req),
            record=record,
            units=units,
        )
        atomic_write_json(self.run_dir(req) / OUTCOME_FILE, result.as_dict())
        return result

    def _refused(self, duel: _Duel, reason: str) -> DuelResult:
        """A refused challenger: nothing runs, nothing is published, and the king is not asked."""
        log.warning("duel %s is refused: %s", duel.event_id[:16], reason)
        for row in duel.units:
            row["challenger_success"], row["challenger_error"] = False, reason
        return self._decided(duel, "refused", reason)

    def _void(self, duel: _Duel, reason: str) -> DuelResult:
        log.warning("duel %s is void: %s", duel.event_id[:16], reason)
        return self._decided(duel, "void", reason)

    def _decided(self, duel: _Duel, status: str, reason: str) -> DuelResult:
        result = DuelResult(
            status=status,
            kind=duel.req.kind,
            event_id=duel.event_id,
            duel_id=duel.duel_id,
            reason=reason,
            run_dir=duel.run_dir,
            units=duel.units,
        )
        atomic_write_json(duel.run_dir / OUTCOME_FILE, {**result.as_dict(), "sides": duel.sides})
        self._post(duel, force=True, phase="failed", side=None, message=f"{status}: {reason}")
        return result

    def _failed(self, duel: _Duel, reason: str) -> None:
        log.error("duel %s failed: %s", duel.event_id[:16], reason)
        try:
            (duel.run_dir / FAILED_FILE).write_text(reason + "\n", encoding="utf-8")
        except OSError:
            pass
        self._post(duel, force=True, phase="failed", side=None, message=f"failed: {reason}")

    # -- plumbing ---------------------------------------------------------------------------

    def _published(self, track: str, eid: str) -> dict[str, Any] | None:
        """The index's record of this event, if an earlier run published it before it stopped."""
        for record in reversed(self.store.iter_index(track)):
            if record.get("event_id") == eid:
                return record
        return None

    def _benchmark_info(self, duel: _Duel) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, plugin in duel.benchmarks.items():
            pin = self.spec.benchmark_pin(name) or {}
            try:
                info = json.loads(json.dumps(plugin.info(), default=str))
            except Exception as exc:  # noqa: BLE001 - recorded, since the duel already ran
                info = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                installed = metadata.version(str(pin.get("distribution")))
            except (metadata.PackageNotFoundError, ValueError):
                installed = None
            out[name] = {"info": info, "pin": pin, "installed_version": installed}
        return out

    def push_touched(self) -> None:
        """What the store wrote since the last push, to the mirror when there is one."""
        if self.mirror is None:
            return
        files = self.store.drain_touched()
        if not files:
            return
        try:
            self.mirror.push(files)
        except Exception as exc:  # noqa: BLE001 - mirroring is retried with the next push
            log.warning("mirror push failed: %s", exc)
            self.store.touched.update(files)

    def _post(self, duel: _Duel, *, force: bool = False, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(duel, key, value)
        req = duel.req
        try:
            frame = build_frame(
                self.spec,
                track=req.track,
                validator_key=self.store.signer.verify_key_hex,  # type: ignore[union-attr]
                event_id=duel.event_id,
                kind=req.kind,
                duel_size=duel.size,
                king=req.king.as_dict() if req.king else None,
                challenger=req.challenger.as_dict(),
                phase=duel.phase,
                side=duel.side,
                units=duel.units,
                current=duel.current,
                recent_media=duel.recent_media,
                message=duel.message,
                started_at=duel.started_at,
            )
        except ValueError:
            log.exception("a live frame could not be built")
            return
        self.live.post(frame, force=force)


def _other(side: str) -> str:
    return "king" if side == "challenger" else "challenger"


def _rate(value: Any) -> float | None:
    """A progress the schema can hold: a number in [0, 1]."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value)
    return None


def _as_king(row: Mapping[str, Any]) -> dict[str, Any]:
    """A genesis row: the entrant ran as the challenger and is published as the king it becomes."""
    out = dict(row)
    for field_ in ("success", "video", "progress", "metric", "steps", "error"):
        out[f"king_{field_}"] = row.get(f"challenger_{field_}")
        out[f"challenger_{field_}"] = None
    out["outcome"] = "tie"
    return out


def _note(kind: str, scoring: Mapping[str, Any]) -> str:
    if kind == "genesis":
        return "The first entrant took the empty throne, scored on its own units."
    delta = scoring.get("delta_points")
    points = "unscored" if delta is None else f"{delta:+.2f} points"
    return (
        f"Crown rule: {scoring['reason']} ({points} against a margin of "
        f"{scoring['score_margin']:g})."
    )


def side_share(budgets: Mapping[str, Any], materialized_s: float, sides: int) -> float:
    """The wall clock each side of a duel may spend: `side_wall_seconds`, but never more than an
    even share of what materializing left of `duel_wall_seconds`. The challenger plays first, and
    without a share it could spend the king's time, voiding the king's last units for both."""
    left = float(budgets["duel_wall_seconds"]) - float(materialized_s)
    return max(0.0, min(float(budgets["side_wall_seconds"]), left / max(1, sides)))


def _work_seconds(run_dir: Path) -> float:
    """The wall time an earlier run of this duel spent materializing and playing units."""
    from .materialize import read_manifest

    spent = sum(p.wall_s for p in read_manifest(run_dir / "prompts").values())
    for side in SIDES:
        spent += sum(float(r.get("wall_s") or 0.0) for r in read_results(run_dir / side).values())
    return spent


def _seconds_since(iso: str) -> float:
    try:
        started = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
