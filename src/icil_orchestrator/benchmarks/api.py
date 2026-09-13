"""What a benchmark distribution must expose, and how the orchestrator checks it.

A benchmark is a separate repository, installed as its own distribution and found through the
`icil.benchmarks` entry point group. This module is the contract between the two, and it is
deliberately one-directional: a benchmark must never import `icil_orchestrator`, because a
benchmark that stands on its own cannot depend on the competition that scores it. So `Benchmark` is
a `Protocol`, nothing is subclassed, and a plugin is checked structurally by `validate_plugin`.

The surface splits in two, and the split is the reason the design works:

`PURE_METHODS` must import and run with **no simulator, no assets and no GPU**. The orchestrator
host, CI and a laptop all read a benchmark's catalogue, derive its unit list and verify a prompt;
none of them can build a scene. A plugin therefore keeps every simulator import function-local, or
better, out of the plugin module altogether.

`COMMAND_METHODS` return an **argv**, not a result. Everything that needs a simulator happens in a
subprocess the orchestrator launches, so the orchestrator never imports SAPIEN or MuJoCo, and the
simulator side can run in another image without a line changing here. The files each command
writes into its `out_dir` are named below (`PROMPT_FILE` and friends).

`api_version` is checked. Version 1 was designed against one benchmark (RoboTwin) and the second
will bend it; bumping it is expected, and is why the number is on the plugin rather than implied.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

#: The ABI this orchestrator speaks. A plugin declaring another number is refused rather than
#: called: the failure modes of a half-matching benchmark are silent and expensive.
BENCHMARK_API_VERSION = 1

#: Where a benchmark distribution advertises itself. The entry point's name is the benchmark id a
#: skill names in `spec.json`; its value is the plugin object, or a module exposing `BENCHMARK`.
ENTRY_POINT_GROUP = "icil.benchmarks"

#: Importable and callable with no simulator, no assets and no GPU.
PURE_METHODS = ("info", "catalogue", "derive_units", "verify_prompt", "read_result")

#: Return the argv of a subprocess that does need those things.
COMMAND_METHODS = ("materialize_command", "run_command")

METHODS = PURE_METHODS + COMMAND_METHODS

#: Keyword parameters each method must accept, so the orchestrator can call it by name.
REQUIRED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "derive_units": ("seed_material", "count", "suite", "category"),
    "verify_prompt": ("path", "unit"),
    "read_result": ("out_dir",),
    "materialize_command": ("unit", "out_dir"),
    "run_command": ("unit", "prompt", "out_dir", "policy_address", "authkey_env"),
}

#: Written by `materialize_command`'s subprocess: the prompt's named arrays (including the
#: privileged `meta`, which never reaches a policy) ...
PROMPT_FILE = "prompt.npz"
#: ... the demonstration clip the dashboard plays ...
DEMONSTRATION_CLIP = "demonstration.mp4"
#: ... and, by both commands, the result `read_result` reads.
RESULT_FILE = "result.json"
#: Written by `run_command`'s subprocess: the rollout clip.
EVALUATION_CLIP = "evaluation.mp4"

#: Keys every unit `derive_units` returns must carry, and the keys its `instance_params` must.
UNIT_KEYS = ("task", "task_label", "instance_params")
INSTANCE_PARAM_KEYS = ("scene_seed", "embodiment")


@runtime_checkable
class Benchmark(Protocol):
    """One benchmark, as the orchestrator sees it."""

    #: Stable identifier, matching the entry point's name and `skills.<skill>.benchmark`.
    id: str
    #: The `BENCHMARK_API_VERSION` this plugin was written against.
    api_version: int

    # -- pure ---------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Identity and shape: id, api_version, embodiment, action space, cameras and the commits
        it is pinned to. Recorded on every duel it runs."""

    def catalogue(self) -> dict[str, Any]:
        """What can be drawn from: `{"suites": {suite: [task, ...]}, "categories": {category:
        label}, "tasks": {task: {"category": ...}}}` and whatever else the benchmark publishes.
        No demonstrations - a catalogue is the menu, not the meal."""

    def derive_units(
        self, *, seed_material: str, count: int, suite: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        """`count` units from `suite` (restricted to `category` when given), a pure function of
        its arguments.

        Only the benchmark knows what a unit of it means - a scene seed, an embodiment - which is
        why derivation lives here and not in the orchestrator. It must be reproducible by a third
        party holding the published record, so derive from a hash of `seed_material`, never from
        a global RNG whose stream depends on a library version. Each unit carries at least
        `UNIT_KEYS`, and its `instance_params` at least `INSTANCE_PARAM_KEYS`.
        """

    def verify_prompt(self, *, path: str, unit: Mapping[str, Any]) -> dict[str, Any]:
        """Check a materialized prompt is the one `unit` asked for, without a simulator.

        Returns at least `{"ok": bool, "sha256": str, "problems": [str, ...]}`. This is what lets
        anyone holding the published prompt confirm it, so it must read the file rather than trust
        a manifest.
        """

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        """One command's result, as its subprocess wrote it in `out_dir`.

        Returns at least `{"success": bool | None, "void": bool, "steps": int | None,
        "error": str | None}`. `success` is None exactly when the unit is void.
        """

    # -- commands -----------------------------------------------------------------------

    def materialize_command(self, *, unit: Mapping[str, Any], out_dir: str) -> Sequence[str]:
        """The argv that writes `PROMPT_FILE`, `DEMONSTRATION_CLIP` and `RESULT_FILE` for one
        unit into `out_dir`. The orchestrator gives it a directory of its own, never a side's."""

    def run_command(
        self,
        *,
        unit: Mapping[str, Any],
        prompt: str,
        out_dir: str,
        policy_address: str,
        authkey_env: str,
        **extra: Any,
    ) -> Sequence[str]:
        """The argv that runs one unit from the prompt at `prompt` against a policy already served
        at `policy_address`, writing `RESULT_FILE` and `EVALUATION_CLIP` into `out_dir`.

        `authkey_env` is the *name* of the environment variable holding the policy's authkey (hex).
        The key itself never appears on a command line, where any process on the host can read it.
        """


def validate_plugin(obj: Any) -> list[str]:
    """Everything wrong with `obj` as a benchmark plugin, in the order a reader would find it.

    Structural, not nominal: a plugin cannot subclass anything of ours without importing us.
    Returns an empty list when the plugin is usable.
    """
    errors: list[str] = []

    identifier = getattr(obj, "id", None)
    if not isinstance(identifier, str) or not identifier:
        errors.append("id: expected a non-empty string")

    version = getattr(obj, "api_version", None)
    if not isinstance(version, int) or isinstance(version, bool):
        errors.append("api_version: expected an integer")
    elif version != BENCHMARK_API_VERSION:
        errors.append(
            f"api_version: speaks {version}, this orchestrator speaks {BENCHMARK_API_VERSION}"
        )

    for name in METHODS:
        method = getattr(obj, name, None)
        if method is None:
            errors.append(f"{name}: missing")
            continue
        if not callable(method):
            errors.append(f"{name}: not callable")
            continue
        errors.extend(_keyword_errors(name, method))

    return errors


def _keyword_errors(name: str, method: Any) -> list[str]:
    """The keyword arguments the orchestrator passes, which `method` must accept by name."""
    required = REQUIRED_KEYWORDS.get(name, ())
    if not required:
        return []
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # a builtin or a C callable: take it on trust
        return []
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    accepted = {
        n
        for n, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return [f"{name}: does not accept {n}" for n in required if n not in accepted]
