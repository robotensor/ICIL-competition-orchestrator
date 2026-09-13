"""Building small published histories for the store tests and the fixture store."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from icil_orchestrator.ids import SubmissionRef, event_id
from icil_orchestrator.spec import load_spec_file
from icil_orchestrator.store.records import (
    duel_event,
    empty_skill_scores,
    index_record,
    media_shas,
    unit_tally,
)
from icil_orchestrator.store.writer import Store

TRACK = "franka_1arm"
FINISHED = "2026-09-13T12:00:00Z"
STARTED = "2026-09-13T11:00:00Z"


def small_spec(spec, tmp_path: Path, lines_per_part: int = 2):
    doc = copy.deepcopy(spec.raw)
    doc["store"]["index_lines_per_part"] = lines_per_part
    path = tmp_path / f"spec-{lines_per_part}.json"
    path.write_text(json.dumps(doc))
    return load_spec_file(path)


def make_record(
    spec,
    kind: str,
    block: int,
    king: SubmissionRef | None,
    challenger: SubmissionRef | None,
    *,
    dethroned: bool = False,
    track: str = TRACK,
    finished_at: str = FINISHED,
) -> dict:
    subject = (challenger or king).key  # type: ignore[union-attr]
    skill = spec.skills(track)[0]
    king_scores = challenger_scores = None
    if kind == "duel":
        king_scores = {**empty_skill_scores(spec.skills(track)), skill: 0.5, "average": 0.5}
        challenger_scores = {**empty_skill_scores(spec.skills(track)), skill: 0.9, "average": 0.9}
    return index_record(
        schema=spec.store["schema"],
        event_id=event_id(kind, track, block, subject),
        kind=kind,
        track=track,
        block=block,
        finished_at=finished_at,
        king=king,
        challenger=challenger,
        king_scores=king_scores,
        challenger_scores=challenger_scores,
        score_margin=spec.score_margin(track),
        dethroned=dethroned,
        new_king=challenger if dethroned else None,
        duel_size="smoke",
    )


def publish(
    store: Store, spec, record: dict, units=None, track: str = TRACK, started_at: str = STARTED
) -> int:
    units = units or []
    record.update(unit_tally(units), media_count=len(media_shas(units)))
    event = duel_event(
        record,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        units=units,
        units_per_skill=spec.units_per_skill(track, "smoke"),
        started_at=started_at,
        wall_seconds=1.5,
        demonstration=spec.demonstration(track),
    )
    store.write_event(track, event)
    return store.append(track, record)
