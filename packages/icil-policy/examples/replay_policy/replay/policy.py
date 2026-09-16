"""The replay policy: open-loop playback of the one demonstration.

It ignores every observation. The k-th call to `act` after `reset` returns the demonstration's
`actions[k]`, and the last action once the demonstration has run out. It is the floor a learned
policy has to beat, and the smallest thing that exercises the whole protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class ReplayPolicy:
    #: The demonstration's `actions` are joint-position targets.
    action_type = "qpos"

    def __init__(self) -> None:
        self._actions: np.ndarray | None = None
        self._step = 0

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        if "actions" not in arrays:
            raise ValueError(f"the demonstration has no 'actions', only {sorted(arrays)}")
        actions = np.array(arrays["actions"], dtype=np.float64)  # a copy: arrays are read-only
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(f"'actions' must be a non-empty (T-1, D) array, not {actions.shape}")
        self._actions = actions
        self._step = 0

    def reset(self, seed: int) -> None:
        self._step = 0

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._actions is None:
            raise RuntimeError("act() before set_demonstration()")
        k = min(self._step, len(self._actions) - 1)
        self._step += 1
        return {"action": self._actions[k]}

    def close(self) -> None:
        self._actions = None
