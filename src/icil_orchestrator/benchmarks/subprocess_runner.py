"""Running a plugged benchmark's units, without importing its simulator.

The ABI splits a benchmark in two: a pure half the orchestrator calls directly, and command builders
that return an **argv**. This module is the other end of the second half - it takes `run_command`'s
argv, runs it, and turns what it wrote back into an `Outcome` a duel scores.

Three properties it holds, each the reason for a piece of the code below:

- **Nothing here imports the benchmark's simulator.** The plugin object is a pure Python object;
  everything needing SAPIEN, assets or a GPU happens in the subprocess.
- **A unit that goes wrong is void, not fatal.** A subprocess that cannot start, crashes, times out,
  writes no `result.json`, or writes something unreadable produces a void unit with the reason on
  it, and the next unit still runs. Whether too many voids invalidate the duel is scoring's call
  (`max_void_fraction`), not this module's.
- **The benchmark says what happened; the orchestrator decides what it means.** `read_result` is
  read for the fields the ABI promises, anything missing takes a default rather than raising, and
  what else the benchmark reported is kept aside in `Outcome.extra`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .api import EVALUATION_CLIP, RESULT_FILE

log = logging.getLogger(__name__)

#: What a result file is read for; the rest is the benchmark's own business, kept in `extra`.
RESULT_FIELDS = ("success", "void", "steps", "error", "progress", "metric")

#: The subprocess's stdout and stderr, in its `out_dir`. Never published: a run log.
LOG_FILE = "benchmark.log"

#: How much of the log a void unit's reason carries.
TAIL_CHARS = 400


@dataclass
class Outcome:
    """One unit's result, with the defaults in one place: a benchmark that reports only what the
    ABI requires still produces a complete record."""

    success: bool | None
    void: bool
    steps: int | None
    error: str | None
    wall_s: float = 0.0
    progress: float | None = None
    metric: float | None = None
    #: The rollout clip the command wrote, when it wrote a non-empty one.
    clip: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def outcome_from(result: Mapping[str, Any], *, wall_s: float = 0.0) -> Outcome:
    """Read a plugin's `read_result` mapping.

    `success is None` exactly when the unit is void - the ABI says so, and the two disagreeing is
    the kind of thing that turns into a wrong score rather than an error, so it is reconciled here
    in favour of void.
    """
    void = bool(result.get("void"))
    success = result.get("success")
    if not isinstance(success, bool):
        void = True
    if void:
        success = None
    error = _str_or_none(result.get("error"))
    if void and error is None:
        error = "the benchmark reported the unit void without a reason"
    return Outcome(
        success=success,
        void=void,
        steps=_int_or_none(result.get("steps")),
        error=error,
        wall_s=wall_s,
        progress=_float_or_none(result.get("progress")),
        metric=_float_or_none(result.get("metric")),
        extra={k: v for k, v in result.items() if k not in RESULT_FIELDS},
    )


def voided(reason: str, *, wall_s: float = 0.0) -> Outcome:
    return Outcome(success=None, void=True, steps=None, error=reason, wall_s=wall_s)


def _finite(value: Any) -> bool:
    """A JSON number that can be scored and signed: `json.loads` accepts NaN and Infinity, which
    `int()` refuses and canonical JSON cannot encode."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int_or_none(value: Any) -> int | None:
    return int(value) if _finite(value) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def read_result_file(out_dir: str | Path, name: str = RESULT_FILE) -> dict[str, Any]:
    """The conventional `read_result` for a benchmark that writes one JSON file."""
    path = Path(out_dir) / name
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    if not isinstance(doc, dict):
        return {"success": None, "void": True, "steps": None, "error": f"unreadable {path.name}"}
    return doc


@dataclass
class Completed:
    """How a subprocess ended. Exactly one of `start_error`, `timed_out` or `returncode` says so."""

    returncode: int | None
    timed_out: bool
    start_error: str | None
    wall_s: float
    log_tail: str


def run_argv(
    argv: list[str], *, env: Mapping[str, str], timeout_s: float, log_path: Path
) -> Completed:
    """Run `argv` to completion or to `timeout_s`, its output going to `log_path`.

    It runs in its own session, and whatever ends the unit - a clean exit, a crash, the timeout, or
    the orchestrator itself being interrupted - the whole process group is killed on the way out: a
    simulator that forked a renderer must not outlive its unit and hold the GPU for the next one.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open(log_path, "wb") as sink:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=dict(env),
                start_new_session=True,
            )
        except OSError as exc:
            return Completed(None, False, str(exc), time.monotonic() - started, "")
        returncode: int | None = None
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # The session is the benchmark's alone (start_new_session), so this reaches only what
            # it started - including children still running after the direct child exited.
            _kill_group(proc)
    return Completed(returncode, timed_out, None, time.monotonic() - started, _tail(log_path))


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if proc.poll() is None:
            proc.kill()
    proc.wait()


def _tail(path: Path, limit: int = TAIL_CHARS) -> str:
    """The end of a log of any size: nothing bounds what a benchmark prints, so only the last few
    kilobytes are ever read."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4 * limit))
            data = fh.read(4 * limit)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()[-limit:]


def run_unit(
    benchmark: Any,
    unit: Mapping[str, Any],
    *,
    prompt: str | None,
    out_dir: str | Path,
    policy_address: str,
    authkey_env: str,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Outcome:
    """One unit against a served policy, start to finish. Every failure is a void outcome, never an
    exception: one bad unit must not lose the rest of the duel."""
    name = str(getattr(benchmark, "id", "benchmark"))
    started = time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    environ = dict(os.environ if env is None else env)

    def void(reason: str) -> Outcome:
        return voided(f"{name}: {reason}", wall_s=round(time.monotonic() - started, 3))

    if not prompt:
        return void("the unit carries no materialized prompt")
    key = environ.get(authkey_env)
    if not key:
        return void(f"no policy authkey in ${authkey_env} for the benchmark subprocess")

    try:
        built = benchmark.run_command(
            unit=unit,
            prompt=str(prompt),
            out_dir=str(out),
            policy_address=policy_address,
            authkey_env=authkey_env,
            **dict(extra or {}),
        )
        argv = [str(a) for a in built]
    except Exception as exc:  # noqa: BLE001 - a plugin's own failure is this unit's, not the duel's
        return void(f"run_command failed: {type(exc).__name__}: {exc}")
    if not argv:
        return void("run_command returned an empty argv")
    # Any process on the host can read another's command line; the key travels by variable name.
    if any(key in arg for arg in argv):
        return void("run_command put the policy authkey on the command line; refusing to run it")
    # What an earlier command left here - a previous attempt, or a materialize command pointed at
    # the same directory - must not be read as this run's result or clip.
    for stale in (RESULT_FILE, EVALUATION_CLIP):
        try:
            (out / stale).unlink(missing_ok=True)
        except OSError as exc:
            return void(f"could not clear a stale {stale}: {exc}")

    done = run_argv(argv, env=environ, timeout_s=timeout_s, log_path=out / LOG_FILE)
    if done.start_error is not None:
        return void(f"could not start the benchmark: {done.start_error}")
    if done.timed_out:
        return void(f"unit exceeded its {timeout_s:g}s budget")
    if done.returncode != 0:
        return void(f"exited {done.returncode}: {done.log_tail}")
    if not (out / RESULT_FILE).is_file():
        return void(f"exited 0 but wrote no {RESULT_FILE}: {done.log_tail}")

    try:
        result = benchmark.read_result(out_dir=str(out))
    except Exception as exc:  # noqa: BLE001
        return void(f"read_result failed: {type(exc).__name__}: {exc}")
    if not isinstance(result, Mapping):
        return void(f"read_result returned {type(result).__name__}, not a mapping")

    try:
        outcome = outcome_from(result, wall_s=round(time.monotonic() - started, 3))
    except Exception as exc:  # noqa: BLE001 - whatever the mapping holds, it is this unit's problem
        return void(f"unusable result: {type(exc).__name__}: {exc}")
    if outcome.void and not outcome.error.startswith(f"{name}: "):  # type: ignore[union-attr]
        outcome.error = f"{name}: {outcome.error}"
    clip = out / EVALUATION_CLIP
    if clip.is_file() and clip.stat().st_size > 0:
        outcome.clip = str(clip)
    return outcome


def run_units(
    benchmark: Any,
    units: Iterable[Mapping[str, Any]],
    *,
    work_root: str | Path,
    policy_address: str,
    authkey_env: str,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    prompt_of: Callable[[Mapping[str, Any]], str | None] = lambda unit: unit.get("prompt"),
    on_outcome: Callable[[Mapping[str, Any], Outcome], None] | None = None,
) -> list[Outcome]:
    """Every unit in order, each in `work_root/<unit_id>`. A void unit is recorded and the next one
    runs; `on_outcome` sees each as it finishes, so progress can be reported and persisted."""
    outcomes: list[Outcome] = []
    for unit in units:
        outcome = run_unit(
            benchmark,
            unit,
            prompt=prompt_of(unit),
            out_dir=Path(work_root) / str(unit["unit_id"]),
            policy_address=policy_address,
            authkey_env=authkey_env,
            timeout_s=timeout_s,
            env=env,
        )
        if outcome.void:
            log.warning("unit %s void: %s", unit["unit_id"], outcome.error)
        if on_outcome is not None:
            on_outcome(unit, outcome)
        outcomes.append(outcome)
    return outcomes
