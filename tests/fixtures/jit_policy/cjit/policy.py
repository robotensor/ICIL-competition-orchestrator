"""A competitor that compiles its action function at run time, as a JIT compiler does.

On its first `act` it writes a tiny C function to the temporary directory ($TMPDIR, else /tmp),
compiles it with `gcc -shared` into its cache ($XDG_CACHE_HOME/cjit, or the manifest's
`build_dir`), loads the shared object with ctypes and answers with what the compiled function
computes from the observation. In the policy sandbox that works where the container may both write
and run code - its scratch tmpfs - and nowhere else.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

#: 3 * obs[i] + i + 0.5, element by element.
SOURCE = """\
void cjit_act(const double *obs, double *out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = 3.0 * obs[i] + (double)i + 0.5;
    }
}
"""
LIBRARY = "cjit.so"

_DOUBLES = ctypes.POINTER(ctypes.c_double)


class CJitPolicy:
    action_type = "qpos"

    def __init__(self, build_dir: str | None = None) -> None:
        self.build_dir = build_dir
        self._act: Any = None

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        pass

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._act is None:
            self._act = self._compile()
        obs = np.ascontiguousarray(observation["qpos"], dtype=np.float64)
        out = np.empty_like(obs)
        self._act(obs.ctypes.data_as(_DOUBLES), out.ctypes.data_as(_DOUBLES), len(obs))
        return {"action": out}

    def _compile(self) -> Any:
        source = Path(tempfile.gettempdir()) / "cjit.c"
        source.write_text(SOURCE)
        if self.build_dir is None:
            cache = Path(os.environ["XDG_CACHE_HOME"]) / "cjit"
        else:
            cache = Path(self.build_dir)
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / LIBRARY
        done = subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", "-o", str(library), str(source)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if done.returncode != 0:
            raise RuntimeError(f"gcc failed ({done.returncode}): {done.stderr.strip()}")
        function = ctypes.CDLL(str(library)).cjit_act
        function.argtypes = (_DOUBLES, _DOUBLES, ctypes.c_int)
        function.restype = None
        return function
