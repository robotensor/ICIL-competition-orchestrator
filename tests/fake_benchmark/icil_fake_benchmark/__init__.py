"""A fake benchmark, packaged the way a real one is: its own distribution, found by entry point.

The pure half below imports the standard library, and numpy inside `verify_prompt` to read a
prompt. The command half is `command.py`, a small script run as a subprocess, which imports
`icil_fake_simulator` - a stand-in for SAPIEN - so a test can check that loading and calling the
plugin never loads the "simulator".

It never imports `icil_orchestrator`, as the ABI requires of every benchmark.

`run_command` reads the unit's `fake_behaviour` (default `policy`: drive the served policy) so a
test can make a unit fail, crash, hang or write nothing, the ways a real benchmark subprocess goes
wrong, and its `fake_void_cause` for `policy_then_void`; `materialize_command` reads its
`fake_materialize` (default `succeed`) the same way.
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
        seed = unit["instance_params"]["scene_seed"]
        if doc.get("scene_seed") != seed:
            problems.append(f"scene_seed {doc.get('scene_seed')} is not the unit's {seed}")
        return {
            "ok": not problems,
            "sha256": hashlib.sha256(data).hexdigest(),
            "problems": problems,
        }

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        try:
            return json.loads((Path(out_dir) / "result.json").read_text())
        except (OSError, ValueError):
            return {"success": None, "void": True, "steps": None, "error": "unreadable result.json"}

    def materialize_command(self, *, unit: Any, out_dir: str) -> list[str]:
        return [
            sys.executable,
            COMMAND,
            "materialize",
            "--out",
            out_dir,
            "--task",
            str(unit["task"]),
            "--scene-seed",
            str(unit["instance_params"]["scene_seed"]),
            "--behaviour",
            str(unit.get("fake_materialize", "succeed")),
        ]

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
        return [
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
            "--void-cause",
            str(unit.get("fake_void_cause", "")),
        ]


BENCHMARK = FakeBenchmark()
