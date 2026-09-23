"""Helpers shared by the vector-policy tests: demonstrations, a probe policy, a served policy."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from vector_policy import wire

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


# -- a served policy ---------------------------------------------------------------------------

AUTHKEY_ENV = "VECTOR_TEST_AUTHKEY"

#: A policy that reports what it saw. What `act` returns is chosen by the observation: an array
#: `returns` holding a mode name as bytes, and an array `fail` makes it raise.
PROBE = """
import json, os, sys, time
from pathlib import Path

import numpy as np

import sibling  # beside the policy: importable only with the repository root on sys.path


class Probe:
    action_type = "ee"

    def __init__(
        self, act_sleep_s=0.0, close_marker=None, broken_init=False, linger=False, broken_close=False
    ):
        if broken_init:
            raise RuntimeError("the probe refuses to be built")
        if linger:  # a thread that would keep a politely exiting interpreter alive forever
            import threading
            threading.Thread(target=time.sleep, args=(3600,), daemon=False).start()
        print("probe: built", flush=True)
        self.act_sleep_s = act_sleep_s
        self.close_marker = close_marker
        self.broken_close = broken_close
        self.seed = None
        self.demo = None
        self.info = None
        self.writeable = None

    def reset(self, seed):
        if seed < 0:
            raise ValueError("the probe wants a non-negative seed")
        self.seed = seed

    def set_demonstration(self, arrays, info):
        self.demo = {name: [value.dtype.str, list(value.shape)] for name, value in arrays.items()}
        self.writeable = any(value.flags.writeable for value in arrays.values())
        self.info = info

    def act(self, observation):
        if "fail" in observation:
            print("probe: failing on purpose", flush=True)
            raise RuntimeError("act failed on purpose")
        time.sleep(self.act_sleep_s)
        mode = bytes(observation["returns"]).decode() if "returns" in observation else "state"
        if mode == "none":
            return None
        if mode == "interrupt":
            raise KeyboardInterrupt("not an Exception: it escapes what catches the policy's errors")
        if mode == "no_action":
            return {"x": np.zeros(1)}
        if mode == "object":
            return {"action": np.array([object()], dtype=object)}
        if mode == "scalar":
            return {"action": np.float64(1.0)}
        if mode == "chunk":
            return {"action": np.ones((4, 7), np.float32)}
        if mode == "list":
            return {"action": [1.0, 2.0]}
        if mode == "ragged":
            return {"action": np.zeros(7), "aux": [[1, 2], [3]]}
        if mode == "tensor":

            class OnTheGpu:
                shape = (7,)

                def __array__(self, *args, **kwargs):
                    raise TypeError("can't convert cuda:0 device type tensor to numpy")

            return {"action": OnTheGpu()}
        state = {
            "cwd": os.getcwd(),
            "path0": sys.path[0],
            "key_in_env": "VECTOR_TEST_AUTHKEY" in os.environ,
            "sibling": sibling.VALUE,
            "seed": self.seed,
            "demo": self.demo,
            "info": self.info,
            "writeable": self.writeable,
            "observation": sorted(observation),
        }
        return {"action": np.zeros(7), "state": np.frombuffer(json.dumps(state).encode(), np.uint8)}

    def close(self):
        if self.broken_close:
            raise RuntimeError("the probe fails to close")
        if self.close_marker:
            Path(self.close_marker).write_text("closed")
"""


def write_repo(root: Path, policy: str = "probe:Probe", kwargs=None, extra=None) -> Path:
    """A competitor repository holding the probe policy; returns its manifest's path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "probe.py").write_text(PROBE)
    (root / "sibling.py").write_text("VALUE = 'from the repository root'\n")
    (root / "broken.py").write_text("raise ImportError('broken on purpose')\n")
    (root / "notapolicy.py").write_text("class NotAPolicy:\n    action_type = 'torque'\n")
    for name, text in (extra or {}).items():
        (root / name).write_text(text)
    lines = ["api: 1", f"policy: {policy}"]
    if kwargs:
        lines.append(f"kwargs: {json.dumps(kwargs)}")
    manifest = root / "policy.yaml"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def call(conn, op, fields=None, arrays=None, timeout=20.0):
    """One raw request and its reply, never waiting forever."""
    wire.send(conn, op, fields, arrays)
    assert conn.poll(timeout), f"no reply to {op} within {timeout}s"
    return wire.recv(conn)


@dataclass
class Served:
    process: subprocess.Popen
    address: str
    authkey: bytes
    log_file: Path

    def connect(self, authkey: bytes | None = None, timeout: float = 20.0):
        family, target = wire.parse_address(self.address)
        deadline = time.monotonic() + timeout
        while True:
            try:
                return Client(target, family=family, authkey=authkey or self.authkey)
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline or self.process.poll() is not None:
                    raise
                time.sleep(0.05)

    def wait(self, timeout: float = 20.0) -> int:
        return self.process.wait(timeout)

    def log(self) -> str:
        return self.log_file.read_text() if self.log_file.exists() else ""


def free_tcp_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{s.getsockname()[1]}"
