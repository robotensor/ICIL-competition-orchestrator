"""A duel's unit list, from the benchmarks behind its track's skills.

Only the benchmark knows what one of its units *is* - a task, a scene seed, an embodiment - so the
ABI puts `derive_units` on the plugin and this module calls it. Two things the orchestrator keeps
for itself, because they are the competition's and not the benchmark's:

- **The identity of a unit.** `unit_id` is `<skill code>-<index>`, indexed across the whole track,
  so a record reads the same whichever benchmark produced it and one benchmark cannot collide with
  another's ids.
- **The seed material.** Derivation must be reproducible by a third party holding the published
  record, so it is a pure function of the duel id and the skill - never of a clock, a global RNG or
  anything the benchmark chooses for itself.

What the benchmark returns beyond that is passed through untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..ids import unit_id, unit_seed
from .api import INSTANCE_PARAM_KEYS

#: Keys the orchestrator sets on every unit. A benchmark returning one has it overwritten rather
#: than honoured: these are the competition's to decide.
RESERVED = ("unit_id", "skill", "index", "seed", "benchmark", "instance", "demo")


class DerivationError(RuntimeError):
    """A benchmark did not return the units its track asked for."""


def seed_key(duel_id: str, entropy: str | None = None) -> str:
    """What a duel's draws hang off: its id, and the chain entropy it was fought under when it has
    one (`DuelRequest.entropy`: a block number and that block's hash). Without entropy anyone can
    compute a duel's units from public values before submitting; with it, not before the block."""
    return duel_id if entropy is None else f"{duel_id}|{entropy}"


def seed_material(duel_id: str, skill: str, entropy: str | None = None) -> str:
    return f"{seed_key(duel_id, entropy)}|{skill}"


def plugin_units(
    spec: Any,
    track: str,
    duel_id: str,
    size: str | None = None,
    *,
    resolve: Callable[[str], Any] | None = None,
    entropy: str | None = None,
) -> list[dict[str, Any]]:
    """Every unit of one duel: skills in spec order, units in the order each benchmark returned
    them, so the list is a pure function of `(spec, track, duel_id, size, entropy)`.

    `resolve` maps a benchmark name to its plugin object; by default the pinned, installed plugin.
    `entropy` is the chain entropy the duel was fought under (`seed_key`), None for none.
    """
    if resolve is None:
        from .plugins import load

        def resolve(name: str) -> Any:
            return load(spec, name)

    loaded: dict[str, Any] = {}
    count = spec.units_per_skill(track, size)
    out: list[dict[str, Any]] = []
    for skill in spec.skills(track):
        name = spec.benchmark_of(skill)
        if name not in loaded:
            loaded[name] = resolve(name)
        benchmark = loaded[name]
        try:
            derived = benchmark.derive_units(
                seed_material=seed_material(duel_id, skill, entropy),
                count=count,
                suite=spec.suite(skill),
                category=spec.category(skill),
            )
        except Exception as exc:  # noqa: BLE001 - named, because a bare traceback here is useless
            raise DerivationError(f"{skill}: {name} could not derive its units: {exc}") from exc

        derived = list(derived or [])
        if len(derived) != count:
            raise DerivationError(
                f"{skill}: {name} returned {len(derived)} units, not the {count} "
                f"{track}/{spec.size_of(track, size)} asks for"
            )
        code = spec.skill_code(skill)
        for index, unit in enumerate(derived):
            check_unit(skill, name, index, unit)
            passed = {k: v for k, v in unit.items() if k not in RESERVED}
            uid = unit_id(code, len(out))
            out.append(
                {
                    **passed,
                    "unit_id": uid,
                    "skill": skill,
                    "benchmark": name,
                    "index": index,
                    "seed": unit_seed(seed_key(duel_id, entropy), skill, index),
                    "task": str(passed["task"]),
                    "task_label": str(passed["task_label"]),
                    # A Same Scene track scores the state it demonstrated, so a unit has exactly
                    # one initial state, and the prompt materialized for it is its demonstration.
                    "instance": 0,
                    "demo": uid,
                    "instance_params": dict(passed["instance_params"]),
                }
            )
    return out


def check_unit(skill: str, name: str, index: int, unit: Any) -> None:
    where = f"{skill}: {name} unit {index}"
    if not isinstance(unit, Mapping):
        raise DerivationError(f"{where} is {type(unit).__name__}, not a mapping")
    for key in ("task", "task_label"):
        if not isinstance(unit.get(key), str) or not unit[key]:
            raise DerivationError(f"{where} has no {key}")
    params = unit.get("instance_params")
    if not isinstance(params, Mapping):
        raise DerivationError(f"{where} has no instance_params mapping")
    missing = [k for k in INSTANCE_PARAM_KEYS if k not in params]
    if missing:
        raise DerivationError(f"{where} instance_params lacks {', '.join(missing)}")
