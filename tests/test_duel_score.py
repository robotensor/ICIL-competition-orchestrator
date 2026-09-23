"""The crown rule, ported from icilval with its tests, over the spec's own skills."""

from __future__ import annotations

from vector_orchestrator.duel.score import (
    SCORE_EPSILON,
    crown_moves,
    paired_outcome,
    skill_scores,
    tally,
    too_void,
    verdict,
    void_fraction,
)
from vector_orchestrator.store.records import unit_tally

SKILLS = ("franka_pick_and_place", "franka_stacking")


def unit(skill, k, c, void=False, i=0):
    return {
        "unit_id": f"{skill[7:9]}-{i:03d}",
        "skill": skill,
        "king_success": k,
        "challenger_success": c,
        "outcome": paired_outcome(k, c),
        "void": void,
    }


def test_paired_outcome():
    assert paired_outcome(True, True) == "tie"
    assert paired_outcome(False, False) == "tie"
    assert paired_outcome(False, True) == "challenger"
    assert paired_outcome(True, False) == "king"
    assert paired_outcome(None, True) == "tie"


def test_skill_scores_are_fractions_that_exclude_void_units():
    units = [
        unit("franka_pick_and_place", True, False),
        unit("franka_pick_and_place", True, True, void=True),
        unit("franka_pick_and_place", False, True),
    ]
    k = skill_scores(units, "king", SKILLS)
    c = skill_scores(units, "challenger", SKILLS)
    assert k["franka_pick_and_place"] == 0.5 and c["franka_pick_and_place"] == 0.5
    assert k["franka_stacking"] is None
    assert k["average"] == 0.5 and c["average"] == 0.5
    assert list(k) == [*SKILLS, "average"]


def test_the_crown_rule_at_its_boundary():
    assert crown_moves(0.60, 0.63, 3.0)
    assert not crown_moves(0.60, 0.63 - 1e-6, 3.0)
    assert crown_moves(0.5, 0.5, 0.0)
    assert not crown_moves(None, 0.9, 3.0)
    assert not crown_moves(0.9, None, 3.0)
    assert SCORE_EPSILON < 1e-6


def test_the_average_is_over_skills_and_the_margin_decides(spec):
    skills = spec.skills("franka_1arm")
    units, i = [], 0
    for skill, (kw, cw) in zip(skills, [(6, 8), (3, 4), (0, 0)], strict=True):
        for n in range(10):
            units.append(unit(skill, n < kw, n < cw, i=i))
            i += 1
    v = verdict(units, spec.score_margin("franka_1arm"), skills)
    assert v.king_scores["average"] == (0.6 + 0.3 + 0.0) / 3
    assert v.challenger_scores["average"] == (0.8 + 0.4 + 0.0) / 3
    assert v.dethroned and v.reason == "margin-met"
    assert round(v.delta_points, 6) == 10.0
    assert (v.tally.wins, v.tally.losses, v.tally.decided, v.tally.ties) == (3, 0, 3, 27)
    assert v.tally.as_dict() == unit_tally(units), "the record's tally is the scorer's"


def test_a_copy_of_the_king_never_moves_the_crown():
    units = [unit(s, n % 2 == 0, n % 2 == 0, i=n) for s in SKILLS for n in range(6)]
    v = verdict(units, 3.0, SKILLS)
    assert v.delta_points == 0.0 and not v.dethroned and v.reason == "short-of-margin"
    assert verdict(units, 0.0, SKILLS).dethroned, "a zero margin crowns an equal challenger"


def test_void_fraction_is_over_every_unit_and_only_above_the_limit_is_too_void():
    units = [unit("franka_stacking", True, True, void=n < 2, i=n) for n in range(10)]
    assert void_fraction(units) == 0.2 and void_fraction([]) == 0.0
    assert not too_void(units, 0.2) and too_void(units, 0.19)
    assert tally(units).void == 2


def test_an_unscored_duel_does_not_move_the_crown():
    units = [unit("franka_stacking", None, None, void=True)]
    v = verdict(units, 3.0, SKILLS)
    assert v.reason == "unscored" and not v.dethroned and v.void_fraction == 1.0
    assert verdict([], 3.0, SKILLS).reason == "no-units"
