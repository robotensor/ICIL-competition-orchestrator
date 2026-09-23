from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from policy_testing import (
    AUTHKEY_ENV,
    EXAMPLES,
    Served,
    observation,
    robotwin_demonstration,
    write_repo,
)


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


@pytest.fixture
def probe_repo(tmp_path):
    """`probe_repo(policy=..., kwargs=...)`: a competitor repository serving the probe policy."""
    return lambda **options: write_repo(tmp_path / "repo", **options)


@pytest.fixture
def serve():
    """`serve(manifest, ...)`: `python -m vector_policy.serve` in a subprocess, listening."""
    started: list[Served] = []
    directories: list[str] = []

    def start(manifest, *, address=None, env=None, args=(), listening=True, authkey=None):
        directory = tempfile.mkdtemp(prefix="vectorp-")  # short: a socket path has ~100 bytes
        directories.append(directory)
        authkey = authkey if authkey is not None else secrets.token_bytes(32)
        address = address or os.path.join(directory, "policy.sock")
        log_file = Path(directory) / "serve.log"
        environ = {**os.environ, AUTHKEY_ENV: authkey.hex(), **(env or {})}
        argv = [sys.executable, "-m", "vector_policy.serve", "--manifest", str(manifest)]
        argv += ["--address", address, "--authkey-env", AUTHKEY_ENV, "--log-file", str(log_file)]
        process = subprocess.Popen([*argv, *args], env=environ, cwd=directory)
        served = Served(process, address, authkey, log_file)
        started.append(served)
        if listening:
            deadline = time.monotonic() + 30
            while "listening on" not in served.log():
                assert process.poll() is None, f"the server exited early:\n{served.log()}"
                assert time.monotonic() < deadline, f"the server never listened:\n{served.log()}"
                time.sleep(0.02)
        return served

    yield start
    for served in started:
        if served.process.poll() is None:
            served.process.kill()
        served.process.wait()
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)
