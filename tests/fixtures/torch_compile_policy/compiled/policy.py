"""A competitor whose `act` runs a torch.compile'd function on the CPU.

Inductor, torch.compile's default backend, generates C++ for the function, builds it with g++ into
$TORCHINDUCTOR_CACHE_DIR and loads the shared object: it needs the base image's toolchain and the
sandbox's executable /tmp. The function is compiled once, in the constructor - within `hello`'s
start budget - for the observation's shape, so each `act` runs compiled code well inside its own
budget. One compile worker is plenty for one small function and gentle on a shared host.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def step(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) * 2.0 + torch.cos(x) ** 2


class CompiledPolicy:
    action_type = "qpos"

    def __init__(self, dim: int = 16) -> None:
        self.compiled = torch.compile(step, backend="inductor", fullgraph=True, dynamic=False)
        started = time.monotonic()
        self.compiled(torch.zeros(dim, dtype=torch.float64))
        self.compile_seconds = time.monotonic() - started

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        pass

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.array(observation["qpos"], dtype=np.float64))
        started = time.monotonic()
        action = self.compiled(x).numpy()
        return {
            "action": action,
            "compile_seconds": np.array(self.compile_seconds),
            "act_seconds": np.array(time.monotonic() - started),
        }
