"""Every unit's prompt, produced once per duel, before either side runs.

A track that scores the very scene it demonstrates (`prompts: "materialized"`) cannot publish its
prompts ahead of time: the demonstration is the answer for that scene. So the orchestrator has the
benchmark produce each unit's prompt here, on this host, before either side starts, and publishes
the prompts' sha256 with the event. Both sides then run from those same files, which is the whole
of what makes a duel fair on RoboTwin: an expert's trajectory is not reproducible from its seed
(CuRobo's IK seed generators live as long as the environment, and planning retries stop on a wall
clock), so a prompt produced per side would differ per side.

For each unit, in order, and resumably:

1. `materialize_command(unit, out_dir)` runs as a subprocess with the allow-listed environment and
   `budgets.materialize_wall_seconds`, in `<root>/<unit_id>/`.
2. Its `result.json` must say the demonstration succeeded, and `prompt.npz` must exist.
3. `verify_prompt(path, unit)` must say ok, and hash the file to what this module hashes it to:
   the published `prompt_sha256` is the sha256 of the file's bytes, which is what a third party
   holding the file checks. A `prompt_sha256` the result names must be that hash too.
4. The scene seed the prompt was built on is kept (`Prompt.scene_seed`): the result's
   `scene_seed`, which must be the one `verify_prompt` read from the file when it names one. A
   benchmark may choose the scene only here - RoboTwin's expert tries a unit's candidate seeds in
   order and keeps the first it solves - so the duel publishes it as the unit's
   `instance_params.scene_seed`, the scene both sides played.

A unit whose materialization fails or is rejected is **void for both sides**, with the reason; it
is not retried, since a second try would be a different demonstration than the one already
recorded. Each unit's prompt is appended to `<root>/manifest.jsonl` as it is done, so a restarted
duel re-uses it rather than producing a new one, after checking the file still has its hash.

Prompts stay in the run directory. The benchmark subprocess reads them from there; no policy
container ever mounts it, and the prompt's privileged `meta` never crosses the policy socket.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..benchmarks.api import DEMONSTRATION_CLIP, PROMPT_FILE, RESULT_FILE
from ..benchmarks.subprocess_runner import benchmark_environment, outcome_from, run_argv
from ..canon import sha256_file
from .orphans import Ledger, reap_ledger

log = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.jsonl"
#: The materialize command's output, in its unit's directory. A run log, never published.
LOG_FILE = "materialize.log"


def scene_seed_of(value: Any) -> int | None:
    """`value` as a scene seed: an integer, never a bool; None for anything else."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class Prompt:
    """One unit's prompt on this host, or why it has none."""

    unit_id: str
    path: Path | None = None
    sha256: str | None = None
    demo_clip: Path | None = None
    demo_sha256: str | None = None
    void: bool = False
    error: str | None = None
    wall_s: float = 0.0
    #: The scene seed the benchmark built the prompt on, when it says.
    scene_seed: int | None = None

    def changed(self) -> str | None:
        """Why the file is no longer the prompt that was recorded, or None while it is."""
        if self.void or self.path is None:
            return None
        try:
            found = sha256_file(self.path)
        except OSError as exc:
            return f"the prompt file cannot be read any more: {exc}"
        if found != self.sha256:
            return f"the prompt file changed after it was materialized ({found[:12]}...)"
        return None

    def published(self) -> dict[str, Any]:
        """The event's `prompts` entry: the unit and the sha256 both sides ran from."""
        return {"unit_id": self.unit_id, "sha256": self.sha256, "demo_video": self.demo_sha256}

    def to_line(self, root: Path) -> dict[str, Any]:
        def rel(path: Path | None) -> str | None:
            return None if path is None else str(path.relative_to(root))

        return {
            "unit_id": self.unit_id,
            "prompt": rel(self.path),
            "sha256": self.sha256,
            "demo_clip": rel(self.demo_clip),
            "demo_sha256": self.demo_sha256,
            "void": self.void,
            "error": self.error,
            "wall_s": self.wall_s,
            "scene_seed": self.scene_seed,
        }

    @classmethod
    def from_line(cls, root: Path, doc: Mapping[str, Any]) -> Prompt:
        def path(value: Any) -> Path | None:
            return root / str(value) if value else None

        return cls(
            unit_id=str(doc["unit_id"]),
            path=path(doc.get("prompt")),
            sha256=doc.get("sha256"),
            demo_clip=path(doc.get("demo_clip")),
            demo_sha256=doc.get("demo_sha256"),
            void=bool(doc.get("void")),
            error=doc.get("error"),
            wall_s=float(doc.get("wall_s") or 0.0),
            scene_seed=scene_seed_of(doc.get("scene_seed")),
        )


@dataclass
class Materialized:
    root: Path
    prompts: dict[str, Prompt]

    def manifest(self) -> list[dict[str, Any]]:
        """Every prompt both sides ran from, in unit order; a void unit has none."""
        return [p.published() for p in self.prompts.values() if not p.void]

    @property
    def void(self) -> int:
        return sum(1 for p in self.prompts.values() if p.void)


def read_manifest(root: Path) -> dict[str, Prompt]:
    """The prompts already produced under `root`, the last line for a unit winning."""
    out: dict[str, Prompt] = {}
    try:
        text = (root / MANIFEST_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        try:
            doc = json.loads(line)
            prompt = Prompt.from_line(root, doc)
        except (ValueError, KeyError, TypeError):
            continue  # a torn final line from a crash: that unit is materialized again
        out[prompt.unit_id] = prompt
    return out


def _append(root: Path, prompt: Prompt) -> None:
    with open(root / MANIFEST_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(prompt.to_line(root), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def materialize_units(
    spec: Any,
    units: Iterable[Mapping[str, Any]],
    root: Path,
    *,
    benchmark_of: Callable[[Mapping[str, Any]], Any],
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    deadline: float | None = None,
    stop: Callable[[], bool] | None = None,
    on_prompt: Callable[[Mapping[str, Any], Prompt], None] | None = None,
) -> Materialized:
    """Every unit's prompt under `root`, re-using what an earlier run of this duel produced.

    `deadline` is the duel's, as a `time.monotonic()`: no unit's materialization runs past it, and
    a unit not started by then has no prompt. `stop`, asked before each unit, ends the loop early
    when it says so (the duel is already certain to be void); the units after it have no entry."""
    root.mkdir(parents=True, exist_ok=True)
    budget = float(spec.budgets["materialize_wall_seconds"]) if timeout_s is None else timeout_s
    environ = dict(benchmark_environment(os.environ, "") if env is None else env)
    done = read_manifest(root)
    prompts: dict[str, Prompt] = {}
    for unit in units:
        if stop is not None and stop():
            break
        unit_id = str(unit["unit_id"])
        prompt = done.get(unit_id)
        if prompt is not None:
            reason = prompt.changed()
            if reason is not None:
                prompt = Prompt(unit_id=unit_id, void=True, error=f"materialize: {reason}")
                _append(root, prompt)
        elif deadline is not None and time.monotonic() >= deadline:
            prompt = Prompt(
                unit_id=unit_id,
                void=True,
                error="materialize: the duel ran out of its wall-clock budget",
            )
            _append(root, prompt)
        else:
            left = budget if deadline is None else min(budget, deadline - time.monotonic())
            prompt = materialize_unit(
                benchmark_of(unit), unit, root / unit_id, env=environ, timeout_s=left
            )
            _append(root, prompt)
        if prompt.void:
            log.warning("unit %s has no prompt: %s", unit_id, prompt.error)
        prompts[unit_id] = prompt
        if on_prompt is not None:
            on_prompt(unit, prompt)
    return Materialized(root=root, prompts=prompts)


def materialize_unit(
    benchmark: Any,
    unit: Mapping[str, Any],
    out_dir: Path,
    *,
    env: Mapping[str, str],
    timeout_s: float,
) -> Prompt:
    """One unit's prompt, produced and verified. Every failure is a void prompt, never a raise."""
    unit_id = str(unit["unit_id"])
    name = str(getattr(benchmark, "id", "benchmark"))
    started = time.monotonic()

    def void(reason: str) -> Prompt:
        return Prompt(
            unit_id=unit_id,
            void=True,
            error=f"{name}: materialize: {reason}",
            wall_s=round(time.monotonic() - started, 3),
        )

    try:
        argv = [str(a) for a in benchmark.materialize_command(unit=unit, out_dir=str(out_dir))]
    except Exception as exc:  # noqa: BLE001 - the plugin's failure is this unit's
        return void(f"materialize_command failed: {type(exc).__name__}: {exc}")
    if not argv:
        return void("materialize_command returned an empty argv")
    out_dir.mkdir(parents=True, exist_ok=True)
    reap_ledger(out_dir)  # an expert a killed orchestrator left running must not write here
    for stale in (PROMPT_FILE, DEMONSTRATION_CLIP, RESULT_FILE):
        try:
            (out_dir / stale).unlink(missing_ok=True)
        except OSError as exc:
            return void(f"could not clear a stale {stale}: {exc}")

    done = run_argv(
        argv, env=env, timeout_s=timeout_s, log_path=out_dir / LOG_FILE, ledger=Ledger(out_dir)
    )
    if done.start_error is not None:
        return void(f"could not start: {done.start_error}")
    if done.timed_out:
        return void(f"exceeded its {timeout_s:g}s budget")
    if done.returncode != 0:
        return void(f"exited {done.returncode}: {done.log_tail}")
    if not (out_dir / RESULT_FILE).is_file():
        return void(f"exited 0 but wrote no {RESULT_FILE}: {done.log_tail}")
    try:
        outcome = outcome_from(benchmark.read_result(out_dir=str(out_dir)))
    except Exception as exc:  # noqa: BLE001
        return void(f"its result could not be read: {type(exc).__name__}: {exc}")
    if outcome.void or not outcome.success:
        return void(f"the demonstration did not succeed: {outcome.error or 'no reason given'}")

    prompt = out_dir / PROMPT_FILE
    if not prompt.is_file():
        return void(f"wrote no {PROMPT_FILE}")
    sha = sha256_file(prompt)
    try:
        verdict = benchmark.verify_prompt(path=str(prompt), unit=unit)
    except Exception as exc:  # noqa: BLE001
        return void(f"verify_prompt failed: {type(exc).__name__}: {exc}")
    if not isinstance(verdict, Mapping) or not verdict.get("ok"):
        problems = verdict.get("problems") if isinstance(verdict, Mapping) else None
        return void(
            "the prompt is not the one the unit asked for: "
            + "; ".join(str(p) for p in (problems or ["no reason given"]))
        )
    if verdict.get("sha256") != sha:
        return void(
            f"verify_prompt hashed the prompt to {str(verdict.get('sha256'))[:12]}..., "
            f"not the sha256 of its bytes {sha[:12]}..."
        )
    written = outcome.extra.get("prompt_sha256")
    if written is not None and written != sha:
        return void(
            f"its result says it wrote a prompt hashing to {str(written)[:12]}..., not the "
            f"sha256 of the file's bytes {sha[:12]}..."
        )
    reported, read = (
        scene_seed_of(outcome.extra.get("scene_seed")),
        scene_seed_of(verdict.get("scene_seed")),
    )
    if reported is not None and read is not None and reported != read:
        return void(
            f"its result says the prompt was built on scene seed {reported}, but the prompt's "
            f"own is {read}"
        )
    # Both sides read this file; neither benchmark run may change it in between.
    os.chmod(prompt, 0o444)
    clip = out_dir / DEMONSTRATION_CLIP
    has_clip = clip.is_file() and clip.stat().st_size > 0
    return Prompt(
        unit_id=unit_id,
        path=prompt,
        sha256=sha,
        demo_clip=clip if has_clip else None,
        demo_sha256=sha256_file(clip) if has_clip else None,
        wall_s=round(time.monotonic() - started, 3),
        scene_seed=reported if reported is not None else read,
    )
