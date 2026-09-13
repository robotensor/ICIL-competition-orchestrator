"""The `docker` command line as the orchestrator calls it, with a script standing in for the
binary: what it is given, what it gets back, and what happens when it does not come back."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from icil_orchestrator.submissions.docker import BuildTimedOut, Docker, DockerError


def fake_binary(path: Path, script: str) -> str:
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_a_build_that_does_not_finish_within_its_timeout_is_killed_and_said_so(tmp_path):
    docker = Docker(binary=fake_binary(tmp_path / "docker", "sleep 30\n"))
    started = time.monotonic()
    with pytest.raises(BuildTimedOut) as info:
        docker.build(tmp_path, "FROM x\n", tag="icil-submission:t", timeout_s=0.5)
    assert time.monotonic() - started < 10, "killed at the timeout, not waited for"
    assert info.value.tag == "icil-submission:t" and info.value.timeout_s == 0.5
    assert str(info.value) == "building icil-submission:t did not finish within 0.5s"
    assert isinstance(info.value, DockerError)


def test_a_build_gets_its_dockerfile_on_stdin_and_reads_the_image_id_back(tmp_path):
    seen = tmp_path / "seen"
    script = (
        f"cat > {seen}.dockerfile\n"
        f'echo "$@" > {seen}.args\n'
        # --iidfile is followed by the path to write the id to.
        'while [ "$#" -gt 0 ]; do if [ "$1" = "--iidfile" ]; then echo -n sha256:'
        + "f" * 64
        + ' > "$2"; fi; shift; done\n'
    )
    docker = Docker(binary=fake_binary(tmp_path / "docker", script))
    image_id = docker.build(
        tmp_path / "ctx", "FROM x\n", tag="t:1", build_args={"A": "1"}, timeout_s=30
    )
    assert image_id == "sha256:" + "f" * 64
    assert (tmp_path / "seen.dockerfile").read_text() == "FROM x\n"
    args = (tmp_path / "seen.args").read_text().split()
    assert args[:6] == ["build", "--tag", "t:1", "--file", "-", "--build-arg"]
    assert args[6] == "A=1" and args[-1] == str(tmp_path / "ctx")


def test_a_missing_binary_is_the_harness_problem(tmp_path):
    with pytest.raises(DockerError, match="not installed or not on PATH"):
        Docker(binary=str(tmp_path / "no-such-docker")).image_id("x")
    assert not os.path.exists(tmp_path / "no-such-docker")
