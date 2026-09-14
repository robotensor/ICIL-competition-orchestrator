"""A fake benchmark, packaged the way a real one is: its own distribution, found by entry point.

The pure half below imports the standard library, and numpy inside `verify_prompt` to read a
prompt. The command half is `command.py`, a small script run as a subprocess, which imports
`icil_fake_simulator` - a stand-in for SAPIEN - so a test can check that loading and calling the
plugin never loads the "simulator".

It never imports `icil_orchestrator`, as the ABI requires of every benchmark.

`run_command` reads the unit's `fake_behaviour` (default `policy`: drive the served policy) so a
test can make a unit fail, crash, hang or write nothing, the ways a real benchmark subprocess goes
wrong, its `fake_void_cause` for `policy_then_void`, and its `fake_step_s`, how long the simulator
takes over each step; `materialize_command` reads its `fake_materialize` (default `succeed`) the
same way. The time limits and the policy log a duel passes in `extra` become flags under the names
the RoboTwin plugin reads them by, and `info()["limits"]` says what the command keeps back from a
unit's timeout.

A unit's `instance_params` name its `scene_seed`, or, as RoboTwin's do, `scene_seeds` to choose
from and no `scene_seed` yet: `materialize` builds on the first candidate (the second under
`fake_materialize: reject_first`) and says which in its result, and `verify_prompt` accepts a
prompt built on any of them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

COMMAND = str(Path(__file__).resolve().parent / "command.py")

CATEGORIES = {
    "pick_and_place": "Pick and Place",
    "stacking": "Stacking",
    "press_push": "Press / Push",
}
TASKS = {
    "place_cube_plate": "pick_and_place",
    "place_cup_coaster": "pick_and_place",
    "stack_two_blocks": "stacking",
    "press_button": "press_push",
}
SUITES = {"franka_1arm": sorted(TASKS)}
EMBODIMENT = ["franka-panda", "franka-panda", 0.6]
#: What `run` keeps back from `--unit-timeout-s` to write its result, as RoboTwin's run-unit does.
RESULT_RESERVE_S = 1.0


def candidate_seeds(unit: Any) -> list[int]:
    """The scene seeds a unit's prompt may be built on: its `scene_seeds`, or its `scene_seed`."""
    params = unit["instance_params"]
    return list(params.get("scene_seeds") or [params["scene_seed"]])


class FakeBenchmark:
    id = "fake"
    api_version = 1

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "api_version": self.api_version,
            "embodiment": EMBODIMENT,
            "cameras": ["head_camera"],
            "action_dims": {"qpos": 16, "ee": 16},
            "limits": {"result_reserve_s": RESULT_RESERVE_S},
        }

    def catalogue(self) -> dict[str, Any]:
        return {
            "suites": {k: list(v) for k, v in SUITES.items()},
            "categories": dict(CATEGORIES),
            "tasks": {t: {"category": c} for t, c in TASKS.items()},
        }

    def derive_units(
        self, *, seed_material: str, count: int, suite: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        if suite not in SUITES:
            raise ValueError(f"no suite {suite!r}")
        tasks = [t for t in SUITES[suite] if category is None or TASKS[t] == category]
        if not tasks:
            raise ValueError(f"suite {suite!r} has no task in category {category!r}")
        units = []
        for i in range(count):
            digest = hashlib.sha256(f"{seed_material}|{i}".encode()).hexdigest()
            task = tasks[int(digest[:8], 16) % len(tasks)]
            units.append(
                {
                    "task": task,
                    "task_label": task.replace("_", " ").capitalize(),
                    "suite": suite,
                    "category": TASKS[task],
                    "instance_params": {
                        "scene_seed": int(digest[8:16], 16),
                        "embodiment": list(EMBODIMENT),
                    },
                }
            )
        return units

    def verify_prompt(self, *, path: str, unit: Any) -> dict[str, Any]:
        import io
        import zipfile

        import numpy as np

        problems = []
        try:
            data = Path(path).read_bytes()
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                doc = json.loads(bytes(arrays["meta"]).decode())
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            return {"ok": False, "sha256": "", "problems": [f"unreadable prompt: {exc}"]}
        if not isinstance(doc, dict):
            return {"ok": False, "sha256": "", "problems": ["the prompt's meta is not an object"]}
        if doc.get("task") != unit["task"]:
            problems.append(f"task {doc.get('task')!r} is not the unit's {unit['task']!r}")
        candidates = candidate_seeds(unit)
        if doc.get("scene_seed") not in candidates:
            problems.append(f"scene_seed {doc.get('scene_seed')} is not the unit's {candidates}")
        return {
            "ok": not problems,
            "sha256": hashlib.sha256(data).hexdigest(),
            "problems": problems,
            "scene_seed": doc.get("scene_seed"),
        }

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        try:
            return json.loads((Path(out_dir) / "result.json").read_text())
        except (OSError, ValueError):
            return {"success": None, "void": True, "steps": None, "error": "unreadable result.json"}

    def materialize_command(self, *, unit: Any, out_dir: str) -> list[str]:
        argv = [
            sys.executable,
            COMMAND,
            "materialize",
            "--out",
            out_dir,
            "--task",
            str(unit["task"]),
        ]
        for seed in candidate_seeds(unit):
            argv += ["--scene-seed", str(seed)]
        return [*argv, "--behaviour", str(unit.get("fake_materialize", "succeed"))]

    def run_command(
        self,
        *,
        unit: Any,
        prompt: str,
        out_dir: str,
        policy_address: str,
        authkey_env: str,
        **extra: Any,
    ) -> list[str]:
        argv = [
            sys.executable,
            COMMAND,
            "run",
            "--prompt",
            prompt,
            "--out",
            out_dir,
            "--policy-address",
            policy_address,
            "--authkey-env",
            authkey_env,
            "--behaviour",
            str(unit.get("fake_behaviour", "policy")),
            "--act-timeout-s",
            str(float(extra.get("act_timeout_s", 30.0))),
            "--step-s",
            str(float(unit.get("fake_step_s", 0.0))),
            "--void-cause",
            str(unit.get("fake_void_cause", "")),
        ]
        for key, flag in (
            ("unit_timeout_s", "--unit-timeout-s"),
            ("policy_budget_s", "--policy-budget-s"),
        ):
            if extra.get(key) is not None:
                argv += [flag, repr(float(extra[key]))]
        if extra.get("policy_log"):
            argv += ["--policy-log", str(extra["policy_log"])]
        return argv


BENCHMARK = FakeBenchmark()
