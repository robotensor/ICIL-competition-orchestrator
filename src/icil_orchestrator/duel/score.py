"""Scoring and the crown rule: pure functions over published unit rows.

The rule, as `spec.duel._crown_comment` states it:

- Per skill, a side's success rate is its successes over the skill's **non-void** units, a
  fraction in [0, 1]; a skill with no scored unit has no rate (None). Its score is the mean of the
  rates it has, under `average`.
- A unit is void for **both** sides when it is void for either: a side that could not be scored on
  a unit must not let the other side be scored on it alone. `void_fraction` is over every unit.
- The challenger takes the crown iff `challenger.average >= king.average + score_margin / 100`.
  `score_margin` is in percentage points and is divided by 100 here and nowhere else.
- A duel whose void fraction exceeds `max_void_fraction` is void and moves nothing.
- Paired per-unit outcomes (who succeeded where the other did not) are published as a diagnostic
  and never decide the crown.

The skill list always comes from the spec, so nothing here names one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SIDES = ("challenger", "king")
#: Averages are means of fractions; a margin met exactly must not be missed to rounding.
SCORE_EPSILON = 1e-9


def paired_outcome(king_success: bool | None, challenger_success: bool | None) -> str:
    """Which side did better on one unit: `challenger`, `king` or `tie` (also when unscored)."""
    if not isinstance(king_success, bool) or not isinstance(challenger_success, bool):
        return "tie"
    if king_success == challenger_success:
        return "tie"
    return "challenger" if challenger_success else "king"


def side_success(unit: dict[str, Any], side: str) -> bool | None:
    value = unit.get(f"{side}_success")
    return value if isinstance(value, bool) else None


def skill_rate(units: Iterable[dict[str, Any]], side: str, skill: str) -> float | None:
    scored = successes = 0
    for u in units:
        if u.get("skill") != skill or u.get("void"):
            continue
        s = side_success(u, side)
        if s is None:
            continue
        scored += 1
        successes += int(s)
    return successes / scored if scored else None


def average(per_skill: dict[str, float | None], skills: Sequence[str]) -> float | None:
    present = [per_skill[s] for s in skills if per_skill.get(s) is not None]
    return sum(present) / len(present) if present else None  # type: ignore[arg-type]


def skill_scores(
    units: Iterable[dict[str, Any]], side: str, skills: Sequence[str]
) -> dict[str, float | None]:
    units = list(units)
    per: dict[str, float | None] = {s: skill_rate(units, side, s) for s in skills}
    per["average"] = average(per, skills)
    return per


def crown_moves(
    king_average: float | None, challenger_average: float | None, margin_points: float
) -> bool:
    if king_average is None or challenger_average is None:
        return False
    return challenger_average >= king_average + margin_points / 100.0 - SCORE_EPSILON


def void_fraction(units: Iterable[dict[str, Any]]) -> float:
    units = list(units)
    return sum(1 for u in units if u.get("void")) / len(units) if units else 0.0


def too_void(units: Iterable[dict[str, Any]], max_void_fraction: float) -> bool:
    return void_fraction(units) > max_void_fraction


@dataclass
class Tally:
    units: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    decided: int = 0
    void: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "decided": self.decided,
            "void": self.void,
        }


def tally(units: Iterable[dict[str, Any]]) -> Tally:
    """The paired diagnostic, from the challenger's side. A void unit counts only as void."""
    t = Tally()
    for u in units:
        t.units += 1
        if u.get("void"):
            t.void += 1
            continue
        outcome = paired_outcome(side_success(u, "king"), side_success(u, "challenger"))
        if outcome == "challenger":
            t.wins += 1
        elif outcome == "king":
            t.losses += 1
        else:
            t.ties += 1
    t.decided = t.wins + t.losses
    return t


@dataclass
class Verdict:
    king_scores: dict[str, float | None]
    challenger_scores: dict[str, float | None]
    score_margin: float
    dethroned: bool
    #: margin-met, short-of-margin, unscored or no-units: why the crown moved or stayed.
    reason: str
    void_fraction: float = 0.0
    tally: Tally = field(default_factory=Tally)

    @property
    def delta_points(self) -> float | None:
        k, c = self.king_scores["average"], self.challenger_scores["average"]
        return None if k is None or c is None else (c - k) * 100.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "score_margin": self.score_margin,
            "delta_points": self.delta_points,
            "void_fraction": self.void_fraction,
        }


def verdict(units: Iterable[dict[str, Any]], score_margin: float, skills: Sequence[str]) -> Verdict:
    """The duel's scores and whether the crown moves. Void units are excluded from both sides."""
    units = list(units)
    king = skill_scores(units, "king", skills)
    challenger = skill_scores(units, "challenger", skills)
    moves = crown_moves(king["average"], challenger["average"], score_margin)
    if not units:
        reason = "no-units"
    elif king["average"] is None or challenger["average"] is None:
        reason = "unscored"
    else:
        reason = "margin-met" if moves else "short-of-margin"
    return Verdict(
        king, challenger, score_margin, moves, reason, void_fraction(units), tally(units)
    )
