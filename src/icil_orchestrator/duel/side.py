"""One side of a duel: every unit in order, a freshly served policy for each, resumable.

Units run in the order the duel lists them - the spec's skills in order, each skill's units in
order. For each one:

1. the prompt it was materialized with is re-hashed (both sides must run from the same bytes);
2. the runtime serves the side's policy for this unit alone (`PolicyRuntime.serve`);
3. the benchmark's `run_command` runs as a subprocess with the allow-listed environment plus the
   policy's key variable, within `min(budgets.unit_wall_seconds, what is left of the side's
   budget)`, and its result is read back as an `Outcome`;
4. the unit's record is appended to `<side_dir>/results.jsonl` and handed to `on_unit`.

A restarted duel reads `results.jsonl` first and runs only the units it does not hold, so a unit
that finished is never run twice; the side's budget counts the wall time already recorded.

What makes a unit void rather than a loss, beyond what `subprocess_runner` already voids:

- its prompt is void (materialization failed) or changed on disk;
- the side's submission was refused (`refused`): every unit, with the reason;
- the policy runtime died: the policy never listened, or died underneath its unit. That unit is
  void, and so is **every remaining unit of the side**, with the first death's reason: a runtime
  that died is not asked to start again forty times over the side's budget;
- the side, or the duel, ran out of wall clock before the unit started.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..benchmarks.subprocess_runner import Outcome, benchmark_environment, run_unit, voided
from ..canon import sha256_file
from .materialize import Materialized
from .runtime import PolicyDied, PolicyRuntime, PreparedSubmission, RuntimeUnavailable

log = logging.getLogger(__name__)

RESULTS_FILE = "results.jsonl"


def read_results(side_dir: Path) -> dict[str, dict[str, Any]]:
    """The units this side has finished, by unit id. A torn final line is a unit not finished."""
    out: dict[str, dict[str, Any]] = {}
    try:
        text = (side_dir / RESULTS_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("unit_id"), str):
            out[record["unit_id"]] = record
    return out


def _append(side_dir: Path, record: dict[str, Any]) -> None:
    side_dir.mkdir(parents=True, exist_ok=True)
    with open(side_dir / RESULTS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def unit_record(
    unit: Mapping[str, Any], outcome: Outcome, side_dir: Path, prompt_sha256: str | None
) -> dict[str, Any]:
    """What `results.jsonl` holds for one unit: the outcome, and its clip by path and sha256."""
    clip = Path(outcome.clip) if outcome.clip and not outcome.void else None
    return {
        "unit_id": str(unit["unit_id"]),
        "skill": unit.get("skill"),
        "success": outcome.success,
        "void": outcome.void,
        "steps": outcome.steps,
        "error": outcome.error,
        "progress": outcome.progress,
        "metric": outcome.metric,
        "wall_s": outcome.wall_s,
        "clip": str(clip.relative_to(side_dir)) if clip else None,
        "clip_sha256": sha256_file(clip) if clip else None,
        "prompt_sha256": prompt_sha256,
    }


def run_side(
    spec: Any,
    *,
    side: str,
    units: Sequence[Mapping[str, Any]],
    prompts: Materialized,
    side_dir: Path,
    benchmark_of: Callable[[Mapping[str, Any]], Any],
    runtime: PolicyRuntime | None,
    prepared: PreparedSubmission | None,
    refused: str | None = None,
    deadline: float | None = None,
    on_start: Callable[[Mapping[str, Any]], None] | None = None,
    on_unit: Callable[[Mapping[str, Any], dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Every unit of one side, by unit id. `deadline` is the duel's, as a `time.monotonic()`."""
    side_dir.mkdir(parents=True, exist_ok=True)
    done = read_results(side_dir)
    budgets = spec.budgets
    spent = sum(float(r.get("wall_s") or 0.0) for r in done.values())
    side_deadline = time.monotonic() + float(budgets["side_wall_seconds"]) - spent
    if deadline is not None:
        side_deadline = min(side_deadline, deadline)
    unit_budget = float(budgets["unit_wall_seconds"])
    extra = {"act_timeout_s": float(budgets["act_timeout_s"])}
    dead: str | None = None
    if prepared is None and refused is None:
        refused = "the submission was not prepared"

    for unit in units:
        unit_id = str(unit["unit_id"])
        if unit_id in done:
            continue
        prompt = prompts.prompts.get(unit_id)
        prompt_sha = prompt.sha256 if prompt is not None and not prompt.void else None
        if refused is not None:
            outcome = voided(f"the {side}'s submission was refused: {refused}")
        elif prompt is None or prompt.void:
            reason = prompt.error if prompt is not None else "no prompt was materialized"
            outcome = voided(f"no prompt: {reason}")
        elif dead is not None:
            outcome = voided(f"the {side}'s policy runtime died earlier in this side: {dead}")
        elif time.monotonic() >= side_deadline:
            outcome = voided("the side ran out of its wall-clock budget")
        elif (changed := prompt.changed()) is not None:
            outcome = voided(f"no prompt: {changed}")
        else:
            if on_start is not None:
                on_start(unit)
            outcome, died = _play(
                runtime,  # type: ignore[arg-type]
                prepared,  # type: ignore[arg-type]
                benchmark_of(unit),
                unit,
                prompt_path=str(prompt.path),
                unit_dir=side_dir / unit_id,
                timeout_s=max(0.0, min(unit_budget, side_deadline - time.monotonic())),
                extra=extra,
            )
            if died is not None:
                dead = f"{unit_id}: {died}"
                outcome = voided(f"the {side}'s policy runtime died: {died}", wall_s=outcome.wall_s)
        if outcome.void:
            log.warning("%s %s void: %s", side, unit_id, outcome.error)
        record = unit_record(unit, outcome, side_dir, prompt_sha)
        _append(side_dir, record)
        done[unit_id] = record
        if on_unit is not None:
            on_unit(unit, record)
    return done


def _play(
    runtime: PolicyRuntime,
    prepared: PreparedSubmission,
    benchmark: Any,
    unit: Mapping[str, Any],
    *,
    prompt_path: str,
    unit_dir: Path,
    timeout_s: float,
    extra: Mapping[str, Any],
) -> tuple[Outcome, str | None]:
    """One unit against a policy served for it: its outcome, and why the policy died, if it did."""
    started = time.monotonic()
    try:
        with runtime.serve(prepared, workdir=unit_dir) as served:
            env = {
                **benchmark_environment(os.environ, served.authkey_env),
                **served.env,
            }
            outcome = run_unit(
                benchmark,
                unit,
                prompt=prompt_path,
                out_dir=unit_dir,
                policy_address=served.address,
                authkey_env=served.authkey_env,
                timeout_s=timeout_s,
                env=env,
                extra=extra,
            )
            return outcome, served.died()
    except (PolicyDied, RuntimeUnavailable) as exc:
        wall = round(time.monotonic() - started, 3)
        return voided(str(exc), wall_s=wall), str(exc)
