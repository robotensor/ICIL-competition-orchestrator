"""Live progress frames: ephemeral, best-effort, never part of the record.

The dashboard's `POST /api/live` checks a frame strictly where a wrong value would misfile it -
schema, validator key, track, event id, phase, side - and silently drops a unit whose skill it does
not know. So `build_frame` refuses to build a frame the dashboard would refuse or quietly thin out,
and the error surfaces here rather than as a 422 nobody reads or a progress bar with rows missing.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from .spec import Spec
from .store.records import now_iso

log = logging.getLogger(__name__)

#: Exactly the phases the dashboard accepts (`PHASES` in its `lib/live/types.ts`). A duel that
#: materializes prompts reports that stage as `checking` until the dashboard knows the phase.
PHASES = ("fetching", "checking", "evaluating", "publishing", "done", "failed")

SIDES = ("challenger", "king")

#: The dashboard's own bounds on the fields it routes by. Matched with `fullmatch`: JavaScript's
#: `$` is the end of the string, Python's also matches before a final newline.
EVENT_ID_RE = re.compile(r"[0-9a-f]{8,64}")
MAX_VALIDATOR_KEY = 128
MAX_MESSAGE = 300


def build_frame(
    spec: Spec,
    *,
    track: str,
    validator_key: str,
    event_id: str,
    kind: str,
    duel_size: str | None,
    king: dict[str, str] | None,
    challenger: dict[str, str] | None,
    phase: str,
    side: str | None,
    units: list[dict[str, Any]],
    current: dict[str, Any] | None,
    recent_media: dict[str, Any] | None,
    message: str,
    started_at: str,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    if track not in spec.tracks:
        raise ValueError(f"track must be one of {spec.tracks}, not {track!r}")
    if not EVENT_ID_RE.fullmatch(event_id):
        raise ValueError(f"event_id must be 8-64 lowercase hex characters, not {event_id!r}")
    if side is not None and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES} or None, not {side!r}")
    if not validator_key.strip() or len(validator_key) > MAX_VALIDATOR_KEY:
        raise ValueError("validator_key must be a non-empty string of at most 128 characters")
    skills = spec.skills(track)
    unknown = sorted({str(u.get("skill")) for u in units} - set(skills))
    if unknown:
        raise ValueError(f"units carry skills {unknown} that track {track} does not score")
    # The dashboard keeps a unit only with a known skill, a non-empty task and a numeric instance,
    # and drops the rest without a word: a progress bar with rows missing is worse than an error.
    for u in units:
        where = f"unit {u.get('unit_id')}"
        if not _text(u.get("task")):
            raise ValueError(f"{where}: task must be a non-empty string, not {u.get('task')!r}")
        if not _number(u.get("instance")):
            raise ValueError(f"{where}: instance must be a number, not {u.get('instance')!r}")
    _check_unit_like("current", current, skills, task_may_be_empty=True)
    if recent_media is not None:
        _check_unit_like("recent_media", recent_media.get("unit"), skills, task_may_be_empty=True)
        if recent_media.get("side") not in SIDES:
            raise ValueError(f"recent_media: side must be one of {SIDES}")
    for name, ref in (("king", king), ("challenger", challenger)):
        if ref is not None and not _text(ref.get("repo")):
            raise ValueError(f"{name}: repo must be a non-empty string")

    per_skill: dict[str, dict[str, dict[str, int]]] = {s: {} for s in SIDES}
    for s in per_skill:
        for skill in skills:
            skill_units = [u for u in units if u.get("skill") == skill]
            done = sum(
                1 for u in skill_units if isinstance(u.get(f"{s}_success"), bool) or u.get("void")
            )
            per_skill[s][skill] = {"done": done, "total": len(skill_units)}
    done = sum(v["done"] for v in per_skill[side].values()) if side in per_skill else 0
    live_units = [
        {
            "unit_id": u.get("unit_id"),
            "skill": u.get("skill"),
            "task": u.get("task"),
            "task_label": u.get("task_label"),
            "instance": u.get("instance"),
            "king_success": u.get("king_success"),
            "challenger_success": u.get("challenger_success"),
            "outcome": u.get("outcome")
            if isinstance(u.get("king_success"), bool)
            and isinstance(u.get("challenger_success"), bool)
            else None,
            "demo_video": u.get("demo_video"),
            "king_video": u.get("king_video"),
            "challenger_video": u.get("challenger_video"),
        }
        for u in units
    ]
    return {
        "schema": int(spec.live["schema"]),
        "validator_key": validator_key,
        "track": track,
        "event_id": event_id,
        "kind": kind,
        "duel_size": duel_size,
        "king": king,
        "challenger": challenger,
        "phase": phase,
        "side": side,
        "done": done,
        "total": len(units),
        "skill_progress": per_skill,
        "current": current,
        "recent_media": recent_media,
        "units": live_units,
        "message": message[:MAX_MESSAGE],
        "started_at": started_at,
        "sent_at": now_iso(),
    }


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _number(value: Any) -> bool:
    """A JSON number, as `typeof x === 'number'` means it: a bool is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_unit_like(
    what: str, unit: Any, skills: Sequence[str], *, task_may_be_empty: bool = False
) -> None:
    """`current` and `recent_media.unit` are rendered only if the dashboard can read all three of
    skill, task and instance; a frame it would null out is not built."""
    if unit is None:
        return
    if not isinstance(unit, Mapping):
        raise ValueError(f"{what}: expected an object, not {unit!r}")
    if unit.get("skill") not in skills:
        raise ValueError(f"{what}: skill {unit.get('skill')!r} is not one this track scores")
    task = unit.get("task")
    if not (isinstance(task, str) if task_may_be_empty else _text(task)):
        raise ValueError(f"{what}: task must be a string, not {task!r}")
    if not _number(unit.get("instance")):
        raise ValueError(f"{what}: instance must be a number, not {unit.get('instance')!r}")


class LiveReporter:
    """POSTs frames to the dashboard. Failures are logged and otherwise ignored: a frame is a
    window onto a duel, and a duel must never stop because the window is shut."""

    def __init__(self, spec: Spec, url: str | None, token: str | None, *, timeout_s: float = 5.0):
        self.spec = spec
        self.url = (url.rstrip("/") + str(spec.live["path"])) if url else None
        self.token = token
        self.timeout_s = timeout_s
        self.min_interval_s = float(spec.live.get("min_interval_s", 1.0))
        self.max_bytes = int(spec.live["max_frame_bytes"])
        self._last_sent = 0.0
        self.sent = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.token)

    def encode(self, frame: dict[str, Any]) -> bytes:
        """The body to POST: the frame, or the frame without its unit list if it is too large."""
        data = json.dumps(frame).encode("utf-8")
        if len(data) > self.max_bytes:
            data = json.dumps({**frame, "units": []}).encode("utf-8")
        return data

    def post(self, frame: dict[str, Any], *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_sent < self.min_interval_s:
            return False
        req = urllib.request.Request(
            self.url,  # type: ignore[arg-type]
            data=self.encode(frame),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - operator-configured url
                resp.read()
            self._last_sent = now
            self.sent += 1
            return True
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.failed += 1
            log.warning("live frame not delivered: %s", exc)
            return False
