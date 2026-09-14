"""One side of a duel: every unit in order, a freshly served policy for each, resumable.

Units run in the order the duel lists them - the spec's skills in order, each skill's units in
order. For each one:

1. the prompt it was materialized with is re-hashed (both sides must run from the same bytes);
2. the runtime serves the side's policy for this unit alone (`PolicyRuntime.serve`);
3. the benchmark's `run_command` runs as a subprocess with the allow-listed environment plus the
   policy's key variable, within `min(budgets.unit_wall_seconds, what is left of the side's
   budget)` less the time the policy took to start, and its result is read back as an `Outcome`.
   `run_command` is given `act_timeout_s`; that subprocess timeout as `unit_timeout_s`, so the
   benchmark can stop calling the policy in time to write why a unit it cannot finish ended;
   `policy_budget_s`, what starting the policy left of `budgets.policy_budget_seconds`, never more
   than that timeout less the benchmark's result reserve (`info()["limits"]["result_reserve_s"]`);
   and `policy_log`, a copy of the end of the policy's log in the unit's directory, which the
   benchmark may quote (`runtime.mirror_log`) - never the log the policy itself writes;
4. whose the outcome is, when it is not a scored one, is decided (`attribute`);
5. the unit's record is appended to `<side_dir>/results.jsonl` and handed to `on_unit`.

A restarted duel reads `results.jsonl` first and runs only the units it does not hold, so a unit
that finished is never run twice; the side's budget counts the wall time already recorded. A unit
whose directory exists without a result was cut off by a kill: the process groups its ledger names
are ended (`orphans`) and its directory moved aside before the unit is played again.

**Whose a unit is.** A unit is void - void for both sides, since the duel merges it so - only for
a cause outside either submission. Anything a side's own submission brings about is that side's
failure on the unit, scored like any failed episode:

- void: its prompt is void (materialization failed) or changed on disk, before the unit or during
  it, or the benchmark reports reading a prompt of another sha256; the side or the duel ran
  out of wall clock before it started; the runtime could not serve (`RuntimeUnavailable`: no
  Docker, a container removed from outside); the benchmark crashed, timed out or wrote nothing
  while the policy was fine; the benchmark said `void_cause: "harness"` (a unit that ran out of
  time while its policy was within its budget), or the policy was ended from outside whatever the
  benchmark said; or the duel already knows the unit is void (`void_units`: void on the other
  side, so not played here only to be thrown away).
- the side's failure: its submission was refused or could not be prepared at all (`refused`: a
  king that forfeits), on every unit that has a prompt; its policy never listened (`PolicyDied`,
  which covers `budgets.policy_start_seconds`) or took all of `budgets.policy_budget_seconds` to
  start; the benchmark said `void_cause: "policy"` (an act timeout, a policy that used up its
  budget, a lost policy); or the benchmark gave no cause and the policy ended badly underneath it
  (a non-zero exit, a signal, running out of memory in its sandbox).

A scored outcome stands whatever the policy did after it: nothing turns a result into a void
afterwards. A policy that died on one unit is served afresh for the next, like any other.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..benchmarks.subprocess_runner import Outcome, benchmark_environment, run_unit, voided
from ..canon import sha256_file
from .materialize import Materialized
from .orphans import Ledger, move_aside, reap_ledger
from .runtime import (
    CAUSES,
    HARNESS,
    POLICY,
    PolicyDied,
    PolicyEnd,
    PolicyRuntime,
    PreparedSubmission,
    RuntimeUnavailable,
    mirror_log,
)

log = logging.getLogger(__name__)

RESULTS_FILE = "results.jsonl"
#: In a unit's directory while its benchmark runs: the end of the policy's log, which the benchmark
#: is given as `policy_log`. Removed with the unit; the whole log is kept as the runtime's log file.
POLICY_LOG_COPY = "policy-tail.log"


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


def failed(reason: str, *, wall_s: float = 0.0) -> Outcome:
    """A unit the side lost by its own doing: a failed episode, with the reason."""
    return Outcome(success=False, void=False, steps=None, error=reason, wall_s=wall_s)


def attribute(outcome: Outcome, end: PolicyEnd | None) -> Outcome:
    """The outcome a side is scored with, given how its policy ended.

    A scored outcome stands. A void one is the side's failure when the benchmark names the policy
    as its cause, or names no cause while the policy ended by its own doing; otherwise it stays
    void. The one thing the orchestrator knows better than the benchmark is a policy ended from
    outside (its container removed, Docker gone): a benchmark cannot tell that from a policy that
    went away, so it is void whatever cause the benchmark gave. What the benchmark reported
    (steps, progress, its clip) is kept either way.
    """
    if not outcome.void:
        if end is not None:
            log.warning("a scored unit's policy ended badly afterwards; the score stands: %s", end)
        return outcome
    cause = outcome.extra.get("void_cause")
    if end is not None and end.cause == HARNESS:
        cause = HARNESS
    elif cause not in CAUSES:
        cause = end.cause if end is not None else HARNESS
    reason = outcome.error or ("the policy failed" if cause == POLICY else "void")
    if end is not None and end.cause == cause:
        reason = f"{reason}\n{end.reason}"
    if cause != POLICY:
        return replace(outcome, error=reason)
    return replace(outcome, success=False, void=False, error=reason)


def result_reserve_s(benchmark: Any) -> float:
    """What a benchmark keeps back from `unit_timeout_s` to write its result, as its
    `info()["limits"]["result_reserve_s"]` says: 0 for one that says nothing readable."""
    try:
        value = benchmark.info()["limits"]["result_reserve_s"]
    except Exception:  # noqa: BLE001 - a plugin that cannot say keeps nothing back
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return max(0.0, float(value))


def unit_limits(
    extra: Mapping[str, Any], *, timeout_s: float, budget_s: float, reserve_s: float
) -> dict[str, Any]:
    """`extra` with the time limits of one unit, as `run_command` takes them.

    `unit_timeout_s` is the subprocess's own timeout. `policy_budget_s` is `budget_s`, what is left
    of the policy's budget, but never more than that timeout less the benchmark's result reserve:
    the benchmark stops calling the policy there whatever its budget says, and a call cut short
    there is the harness's. A benchmark refuses a limit that is not positive, so none is passed.
    """
    limits = dict(extra)
    if timeout_s > 0:
        limits["unit_timeout_s"] = timeout_s
    budget = min(budget_s, timeout_s - reserve_s)
    if budget > 0:
        limits["policy_budget_s"] = budget
    return limits


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
    budget_s: float | None = None,
    void_units: Mapping[str, str] | None = None,
    stop: Callable[[], bool] | None = None,
    on_start: Callable[[Mapping[str, Any]], None] | None = None,
    on_unit: Callable[[Mapping[str, Any], dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Every unit of one side, by unit id. `deadline` is the duel's, as a `time.monotonic()`;
    `budget_s` is the side's wall clock (`budgets.side_wall_seconds` unless the duel gives it
    less); `void_units` maps the units already void for the duel to the reason. `stop`, asked
    before each unit not yet finished, ends the side there when it says so (the duel is already
    certain to be void): nothing more is played for a result that cannot stand."""
    side_dir.mkdir(parents=True, exist_ok=True)
    done = read_results(side_dir)
    budgets = spec.budgets
    spent = sum(float(r.get("wall_s") or 0.0) for r in done.values())
    side_budget = float(budgets["side_wall_seconds"]) if budget_s is None else float(budget_s)
    side_deadline = time.monotonic() + side_budget - spent
    if deadline is not None:
        side_deadline = min(side_deadline, deadline)
    unit_budget = float(budgets["unit_wall_seconds"])
    policy_budget = float(budgets["policy_budget_seconds"])
    extra = {"act_timeout_s": float(budgets["act_timeout_s"])}
    #: Each benchmark's result reserve, asked once per side.
    reserves: dict[str, float] = {}
    if prepared is None and refused is None:
        refused = "the submission was not prepared"

    for unit in units:
        unit_id = str(unit["unit_id"])
        if unit_id in done:
            continue
        if stop is not None and stop():
            break
        prompt = prompts.prompts.get(unit_id)
        prompt_sha = prompt.sha256 if prompt is not None and not prompt.void else None
        if prompt is None or prompt.void:
            reason = prompt.error if prompt is not None else "no prompt was materialized"
            outcome = voided(f"no prompt: {reason}")
        elif void_units and unit_id in void_units:
            outcome = voided(f"not played: {void_units[unit_id]}")
        elif refused is not None:
            outcome = failed(f"the {side}'s submission was refused: {refused}")
        elif time.monotonic() >= side_deadline:
            outcome = voided("the side ran out of its wall-clock budget")
        elif (changed := prompt.changed()) is not None:
            outcome = voided(f"no prompt: {changed}")
        else:
            unit_dir = side_dir / unit_id
            if unit_dir.exists():
                # An attempt that never recorded its result: its orchestrator was killed. What it
                # left running is ended, and what it wrote is kept apart from this attempt.
                reap_ledger(unit_dir)
                moved = move_aside(unit_dir, "interrupted")
                log.warning("%s %s: an interrupted attempt is kept at %s", side, unit_id, moved)
            if on_start is not None:
                on_start(unit)
            benchmark = benchmark_of(unit)
            name = str(getattr(benchmark, "id", ""))
            if name not in reserves:
                reserves[name] = result_reserve_s(benchmark)
            outcome = _play(
                runtime,  # type: ignore[arg-type]
                prepared,  # type: ignore[arg-type]
                benchmark,
                unit,
                side=side,
                prompt_path=str(prompt.path),
                unit_dir=side_dir / unit_id,
                deadline=min(time.monotonic() + unit_budget, side_deadline),
                extra=extra,
                policy_budget_s=policy_budget,
                result_reserve_s=reserves[name],
            )
            # The unit counts only if it ran from the recorded bytes: the file must still hash to
            # them, and so must what the benchmark says it read, whenever it says anything - a
            # value that is not the recorded hex, of whatever type, is another prompt's.
            reported = outcome.extra.get("prompt_sha256")
            if (changed := prompt.changed()) is not None:
                outcome = voided(f"no prompt: {changed} during the unit", wall_s=outcome.wall_s)
            elif reported is not None and reported != prompt.sha256:
                outcome = voided(
                    f"no prompt: the benchmark read a prompt hashing to {str(reported)[:12]}..., "
                    f"not the recorded {str(prompt.sha256)[:12]}...",
                    wall_s=outcome.wall_s,
                )
        if outcome.void:
            log.warning("%s %s void: %s", side, unit_id, outcome.error)
        elif outcome.success is False and outcome.error:
            log.info("%s %s failed: %s", side, unit_id, outcome.error)
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
    side: str,
    prompt_path: str,
    unit_dir: Path,
    deadline: float,
    extra: Mapping[str, Any],
    policy_budget_s: float,
    result_reserve_s: float,
) -> Outcome:
    """One unit against a policy served for it, attributed to whoever ended it.

    `deadline` (a `time.monotonic()`) bounds the whole of it: the benchmark gets only what starting
    the policy left, and the wall time recorded - which a resumed side's budget counts - is all of
    it, starting the policy and waiting for it to end included. Starting the policy counts against
    `policy_budget_s` as well, so a policy cannot spend on starting the time its budget would have
    kept the harness: one that took all of it to start has failed the unit, which is not played."""
    started = time.monotonic()
    end: PolicyEnd | None = None
    try:
        with runtime.serve(prepared, workdir=unit_dir) as served:
            start_s = time.monotonic() - started
            if start_s >= policy_budget_s:
                outcome = failed(
                    f"the {side}'s policy took {start_s:.1f}s to start, its whole budget of "
                    f"{policy_budget_s:g}s for the unit"
                )
            else:
                env = {
                    **benchmark_environment(os.environ, served.authkey_env),
                    **served.env,
                }
                copied = (
                    nullcontext(None)
                    if served.live_log is None
                    else mirror_log(served.live_log, unit_dir / POLICY_LOG_COPY)
                )
                with copied as policy_log:
                    timeout_s = max(0.0, deadline - time.monotonic())
                    limits = unit_limits(
                        extra,
                        timeout_s=timeout_s,
                        budget_s=policy_budget_s - start_s,
                        reserve_s=result_reserve_s,
                    )
                    if policy_log is not None:
                        limits["policy_log"] = str(policy_log)
                    outcome = run_unit(
                        benchmark,
                        unit,
                        prompt=prompt_path,
                        out_dir=unit_dir,
                        policy_address=served.address,
                        authkey_env=served.authkey_env,
                        timeout_s=timeout_s,
                        env=env,
                        extra=limits,
                        ledger=Ledger(unit_dir),
                    )
                end = served.died()
    except PolicyDied as exc:
        wall = round(time.monotonic() - started, 3)
        return failed(f"the {side}'s policy did not start: {exc}", wall_s=wall)
    except RuntimeUnavailable as exc:
        wall = round(time.monotonic() - started, 3)
        return voided(f"the {side}'s policy could not be served: {exc}", wall_s=wall)
    return replace(attribute(outcome, end), wall_s=round(time.monotonic() - started, 3))
