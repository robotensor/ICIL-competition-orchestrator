"""The `docker` command line as the orchestrator calls it, with a script standing in for the
binary: what it is given, what it gets back, and what happens when it does not come back."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from vector_orchestrator.submissions.docker import (
    BuildTimedOut,
    ContainerState,
    Docker,
    DockerError,
)


def fake_binary(path: Path, script: str) -> str:
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_a_build_that_does_not_finish_within_its_timeout_is_killed_and_said_so(tmp_path):
    docker = Docker(binary=fake_binary(tmp_path / "docker", "sleep 30\n"))
    started = time.monotonic()
    with pytest.raises(BuildTimedOut) as info:
        docker.build(tmp_path, "FROM x\n", tag="vector-submission:t", timeout_s=0.5)
    assert time.monotonic() - started < 10, "killed at the timeout, not waited for"
    assert info.value.tag == "vector-submission:t" and info.value.timeout_s == 0.5
    assert str(info.value) == "building vector-submission:t did not finish within 0.5s"
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
    assert "--no-cache" not in args, "Docker's cache is used unless asked otherwise"
    docker.build(tmp_path / "ctx", "FROM x\n", tag="t:2", timeout_s=30, no_cache=True)
    args = (tmp_path / "seen.args").read_text().split()
    assert args[:6] == ["build", "--tag", "t:2", "--file", "-", "--no-cache"]


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
    monkeypatch.setenv("VECTOR_LIVE_TOKEN", "live_secret")
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/docker.sock")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    seen = tmp_path / "env"
    docker = Docker(binary=fake_binary(tmp_path / "docker", f'env > "{seen}.$1"\necho id\n'))
    assert "HF_TOKEN" not in docker.environ and "VECTOR_LIVE_TOKEN" not in docker.environ
    assert docker.environ["DOCKER_HOST"] == "unix:///run/docker.sock"

    docker.run(["--name", "vector-policy-env", "img"], env={"VECTOR_POLICY_AUTHKEY": "ab" * 16})
    docker.image_id("img")
    ran = dict(line.split("=", 1) for line in (tmp_path / "env.run").read_text().splitlines())
    inspected = (tmp_path / "env.image").read_text()
    for secret in ("hf_publish_secret", "live_secret"):
        assert secret not in ran.values() and secret not in inspected
    assert ran["VECTOR_POLICY_AUTHKEY"] == "ab" * 16 and ran["DOCKER_HOST"] and ran["LC_ALL"]
    assert "VECTOR_POLICY_AUTHKEY" not in inspected, "the authkey is for the run it was given to"

    # What is given explicitly is filtered the same way.
    given = Docker(
        environ={"HF_TOKEN": "x", "VECTOR_LIVE_TOKEN": "y", "DOCKER_HOST": "h", "PATH": "p"}
    )
    assert given.environ == {"DOCKER_HOST": "h", "PATH": "p"}


def test_policy_containers_are_listed_with_their_labels_and_a_vanished_one_skipped(tmp_path):
    script = (
        'if [ "$1" = ps ]; then echo "$@" > ' + str(tmp_path / "ps") + "; printf 'aaa\\nbbb\\n'; "
        "exit 0; fi\n"
        "cat <<'JSON'\n"
        '{"name": "/vector-policy-one", "running": true, "labels": '
        '{"vector.orchestrator": "policy", "vector.shared-dir": "/work/a b,c/policy"}}\n'
        "JSON\n"
        "echo 'Error: No such container: bbb' >&2\nexit 1\n"
    )
    docker = Docker(binary=fake_binary(tmp_path / "docker", script))
    (found,) = docker.policy_containers()
    assert found.name == "vector-policy-one" and found.running
    assert found.labels["vector.shared-dir"] == "/work/a b,c/policy", "a path is taken whole"
    assert "label=vector.orchestrator=policy" in (tmp_path / "ps").read_text()


def test_a_containers_state_says_its_exit_its_error_and_whether_it_ran_out_of_memory(tmp_path):
    seen = tmp_path / "format"
    script = (
        f'echo "$5" > {seen}\n'
        'case "$6" in\n'
        '  vector-policy-oom) echo "false true 137 " ;;\n'
        '  vector-policy-live) echo "true false 0 " ;;\n'
        "  vector-policy-err) echo 'false false 127 exec: \"python\": not found' ;;\n"
        '  *) echo "Error: No such container: $6" >&2; exit 1 ;;\n'
        "esac\n"
    )
    docker = Docker(binary=fake_binary(tmp_path / "docker", script))
    assert docker.state("vector-policy-oom") == ContainerState(False, 137, "", oom_killed=True)
    assert "{{.State.OOMKilled}}" in seen.read_text()
    assert docker.state("vector-policy-live") == ContainerState(True, 0, "")
    err = docker.state("vector-policy-err")
    assert (err.exit_code, err.error, err.oom_killed) == (127, 'exec: "python": not found', False)
    gone = docker.state("vector-policy-gone")
    assert (gone.running, gone.exit_code, gone.oom_killed) == (False, None, False)
    assert gone.error.startswith("no such container: Error: No such container")


def test_images_are_listed_by_repository_and_removed_unforced_with_the_reason(tmp_path):
    script = (
        'if [ "$1" = images ]; then printf "vector-submission:k-s\\nvector-submission:<none>\\n"; '
        "exit 0; fi\n"
        'echo "$@" >> ' + str(tmp_path / "rmi") + "\n"
        'if [ "$2" = vector-submission:used ]; then echo "conflict: in use" >&2; exit 1; fi\n'
    )
    docker = Docker(binary=fake_binary(tmp_path / "docker", script))
    assert docker.image_refs("vector-submission") == ["vector-submission:k-s"]
    assert docker.remove_image("vector-submission:k-s", force=False) == ""
    assert docker.remove_image("vector-submission:used", force=False) == "conflict: in use"
    assert docker.remove_image("x:y") == ""
    assert (tmp_path / "rmi").read_text().splitlines() == [
        "rmi vector-submission:k-s",
        "rmi vector-submission:used",
        "rmi --force x:y",
    ]
