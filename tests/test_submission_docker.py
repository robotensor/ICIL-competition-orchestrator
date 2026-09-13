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


def test_the_docker_client_gets_the_allow_list_and_the_authkey_only_when_it_runs(
    tmp_path, monkeypatch
):
    """Nothing the orchestrator holds reaches a docker command, and so a container or a build,
    unless it is passed on purpose: no Hub token, no live token, and the policy's authkey only for
    the `run` it was given to."""
    monkeypatch.setenv("HF_TOKEN", "hf_publish_secret")
    monkeypatch.setenv("ICIL_LIVE_TOKEN", "live_secret")
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/docker.sock")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    seen = tmp_path / "env"
    docker = Docker(binary=fake_binary(tmp_path / "docker", f'env > "{seen}.$1"\necho id\n'))
    assert "HF_TOKEN" not in docker.environ and "ICIL_LIVE_TOKEN" not in docker.environ
    assert docker.environ["DOCKER_HOST"] == "unix:///run/docker.sock"

    docker.run(["--name", "icil-policy-env", "img"], env={"ICIL_POLICY_AUTHKEY": "ab" * 16})
    docker.image_id("img")
    ran = dict(line.split("=", 1) for line in (tmp_path / "env.run").read_text().splitlines())
    inspected = (tmp_path / "env.image").read_text()
    for secret in ("hf_publish_secret", "live_secret"):
        assert secret not in ran.values() and secret not in inspected
    assert ran["ICIL_POLICY_AUTHKEY"] == "ab" * 16 and ran["DOCKER_HOST"] and ran["LC_ALL"]
    assert "ICIL_POLICY_AUTHKEY" not in inspected, "the authkey is for the run it was given to"

    # What is given explicitly is filtered the same way.
    given = Docker(
        environ={"HF_TOKEN": "x", "ICIL_LIVE_TOKEN": "y", "DOCKER_HOST": "h", "PATH": "p"}
    )
    assert given.environ == {"DOCKER_HOST": "h", "PATH": "p"}


def test_policy_containers_are_listed_with_their_labels_and_a_vanished_one_skipped(tmp_path):
    script = (
        'if [ "$1" = ps ]; then echo "$@" > ' + str(tmp_path / "ps") + "; printf 'aaa\\nbbb\\n'; "
        "exit 0; fi\n"
        "cat <<'JSON'\n"
        '{"name": "/icil-policy-one", "running": true, "labels": '
        '{"icil.orchestrator": "policy", "icil.shared-dir": "/work/a b,c/policy"}}\n'
        "JSON\n"
        "echo 'Error: No such container: bbb' >&2\nexit 1\n"
    )
    docker = Docker(binary=fake_binary(tmp_path / "docker", script))
    (found,) = docker.policy_containers()
    assert found.name == "icil-policy-one" and found.running
    assert found.labels["icil.shared-dir"] == "/work/a b,c/policy", "a path is taken whole"
    assert "label=icil.orchestrator=policy" in (tmp_path / "ps").read_text()


def test_images_are_listed_by_repository_and_removed_unforced_with_the_reason(tmp_path):
    script = (
        'if [ "$1" = images ]; then printf "icil-submission:k-s\\nicil-submission:<none>\\n"; '
        "exit 0; fi\n"
        'echo "$@" >> ' + str(tmp_path / "rmi") + "\n"
        'if [ "$2" = icil-submission:used ]; then echo "conflict: in use" >&2; exit 1; fi\n'
    )
    docker = Docker(binary=fake_binary(tmp_path / "docker", script))
    assert docker.image_refs("icil-submission") == ["icil-submission:k-s"]
    assert docker.remove_image("icil-submission:k-s", force=False) == ""
    assert docker.remove_image("icil-submission:used", force=False) == "conflict: in use"
    assert docker.remove_image("x:y") == ""
    assert (tmp_path / "rmi").read_text().splitlines() == [
        "rmi icil-submission:k-s",
        "rmi icil-submission:used",
        "rmi --force x:y",
    ]
