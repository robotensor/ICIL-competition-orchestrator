from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def robotwin_demonstration(steps: int = 6, dims: int = 16, cameras=("head", "left_wrist")):
    """A prompt shaped like RoboTwin's, minus `meta`, which never reaches a policy.

    Dual Franka: `qpos` and `endpose` are 16-D, and there is one action fewer than frames.
    """
    rng = np.random.default_rng(7)
    arrays = {
        f"frames_{camera}": rng.integers(0, 255, (steps, 8, 10, 3), np.uint8) for camera in cameras
    }
    arrays.update(
        qpos=rng.standard_normal((steps, dims)),
        endpose=rng.standard_normal((steps, 16)),
        actions=rng.standard_normal((steps - 1, dims)),
        times=np.arange(steps, dtype=np.float64) / 15.0,
        frequency=np.array(15.0),
    )
    info = {
        "frequency": 15.0,
        "cameras": list(cameras),
        "embodiment": "franka-panda+franka-panda",
        "action_dims": {"qpos": dims, "ee": 16},
    }
    return arrays, info


def observation(arrays, t: int = 0):
    """One observation at step `t` of a demonstration: its frames, `qpos` and `endpose`."""
    return {
        name: value[t]
        for name, value in arrays.items()
        if name.startswith("frames_") or name in ("qpos", "endpose")
    }


@pytest.fixture
def demonstration():
    """`(arrays, info)` of a small RoboTwin-shaped demonstration."""
    return robotwin_demonstration()


@pytest.fixture
def observe():
    """`observe(arrays, t)`: the observation at step `t` of a demonstration."""
    return observation


@pytest.fixture
def examples():
    """The directory holding the example competitor repositories."""
    return EXAMPLES
