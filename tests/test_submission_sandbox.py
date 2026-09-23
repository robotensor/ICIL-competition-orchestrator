"""A submission's container, with Docker stood in for.

`FakeDocker.run` starts the real `python -m vector_policy.serve` on the host in the container's
place, so what is tested is the orchestrator's side of the sandbox: the arguments it would give
`docker run`, the socket directory, the authkey, the wait for `hello` and the removal - not the
kernel's isolation, which the container tests check for real.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from submission_helpers import FAKE_BASE_DIGEST, SHA_A, FakeDocker, write_policy_repo
from vector_orchestrator.ids import SubmissionRef
from vector_orchestrator.submissions import SubmissionRejected
from vector_orchestrator.submissions.checks import check_repository
from vector_orchestrator.submissions.container import (
    AUTHKEY_ENV,
    CACHE_ENV,
    HARDENING,
    PREPARE_HOME,
    SHARED_DIR_BYTES,
    SHARED_DIR_INODES,
    SHARED_DIR_SOURCE,
    TMPFS_HARDENING,
    Owner,
    PolicyContainer,
    can_bound_shared_dir,
    container_argv,
    is_socket,
    mount_shared_dir,
    policy_environment,
    prepare_socket_dir,
    reap_orphans,
    run_argv,
    serve_argv,
    tmpfs_options,
)
from vector_orchestrator.submissions.docker import DockerError
from vector_orchestrator.submissions.image import (
    base_image,
    build_submission_image,
    sandbox_user,
)


@pytest.fixture
def docker():
    fake = FakeDocker()
    yield fake
    fake.kill_all()


@pytest.fixture
def base(spec):
    return base_image(spec, FAKE_BASE_DIGEST)


@pytest.fixture
def ref():
    return SubmissionRef.resolved("org/policy", SHA_A)


def ended_process_owner() -> Owner:
    """The `Owner` of a process that has come and gone."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        stat_line = open(f"/proc/{child.pid}/stat").read()
        start = stat_line.rpartition(")")[2].split()[19]
    finally:
        child.kill()
        child.wait()
    return Owner(child.pid, start, os.readlink("/proc/self/ns/pid"))


# -- the container ------------------------------------------------------------------------------


def test_run_argv_is_exactly_the_specs_sandbox_and_one_shared_directory(spec, tmp_path):
    sandbox = spec.submission["sandbox"]
    owner = Owner(pid=4321, start="98765", pidns="pid:[4026531836]")
    args = run_argv(
        spec,
        image="vector-submission:k-s",
        name="vector-policy-x",
        socket_dir=tmp_path,
        owner=owner,
    )
    assert args == [
        "--detach",
        "--name",
        "vector-policy-x",
        "--label",
        "vector.orchestrator=policy",
        "--label",
        "vector.owner.pid=4321",
        "--label",
        "vector.owner.start=98765",
        "--label",
        "vector.owner.pidns=pid:[4026531836]",
        "--label",
        f"vector.shared-dir={tmp_path}",
        "--network",
        sandbox["network"],
        "--read-only",
        *[
            flag
            for path in sandbox["tmpfs"]
            for flag in ("--tmpfs", f"{path}:exec,nosuid,nodev,size={sandbox['tmpfs_bytes']}")
        ],
        "--user",
        sandbox["user"],
        "--gpus",
        str(sandbox["gpus"]),
        "--memory",
        str(sandbox["memory_bytes"]),
        "--memory-swap",
        str(sandbox["memory_bytes"]),  # equal: no swap, the spec's bytes are the total
        "--cpus",
        str(sandbox["cpus"]),
        "--pids-limit",
        str(sandbox["pids"]),
        *HARDENING,
        "--mount",
        f"type=bind,src={tmp_path},dst=/run/vector",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/home/.cache",
        "--env",
        "TRITON_CACHE_DIR=/tmp/home/.cache/triton",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/home/.cache/torchinductor",
        "--env",
        "TORCH_EXTENSIONS_DIR=/tmp/home/.cache/torch_extensions",
        "--env",
        AUTHKEY_ENV,
        "vector-submission:k-s",
        "sh",
        "-c",
        PREPARE_HOME,
        "sh",
        *serve_argv(spec),
    ]
    assert sandbox["network"] == "none" and sandbox["read_only_root"] is True
    assert sandbox["tmpfs"] == ["/tmp"] and sandbox["tmpfs_exec"] is True
    assert args.count("--mount") == 1 and "--volume" not in args and "-v" not in args
    assert HARDENING == ("--cap-drop", "ALL", "--security-opt", "no-new-privileges")
    assert not {"--cap-add", "--privileged", "--device", "--security-opt=seccomp=unconfined"} & set(
        args
    ), "the relaxation is the tmpfs's exec, and nothing else"
    assert serve_argv(spec) == [
        "python",
        "-m",
        "vector_policy.serve",
        "--manifest",
        f"/submission/{spec.submission['manifest']}",
        "--address",
        "/run/vector/policy.sock",
        "--authkey-env",
        AUTHKEY_ENV,
        "--log-file",
        "/run/vector/policy.log",
    ]
    # A policy that needs no GPU gets none; the spec's count is the default.
    without = run_argv(spec, image="i", name="n", socket_dir=tmp_path, gpus=0)
    assert "--gpus" not in without and without.count("--memory") == 1


def test_the_scratch_tmpfs_options_and_the_cache_variables_come_from_the_spec(
    spec_doc, write_spec, tmp_path
):
    """Whether /tmp runs code and how big it is are the spec's; nosuid and nodev are not, and
    every cache variable points under the first tmpfs path, whatever it is."""
    spec = write_spec(spec_doc)
    sandbox = spec.submission["sandbox"]
    assert tmpfs_options(spec) == f"exec,nosuid,nodev,size={sandbox['tmpfs_bytes']}"
    assert TMPFS_HARDENING == ("nosuid", "nodev")
    doc = json.loads(json.dumps(spec_doc))
    doc["submission"]["sandbox"].update(
        tmpfs=["/scratch", "/var/tmp"], tmpfs_exec=False, tmpfs_bytes=1 << 30
    )
    closed = write_spec(doc, name="noexec.json")
    assert tmpfs_options(closed) == f"noexec,nosuid,nodev,size={1 << 30}"
    args = run_argv(closed, image="i", name="n", socket_dir=tmp_path)
    assert [args[i + 1] for i, a in enumerate(args) if a == "--tmpfs"] == [
        f"/scratch:noexec,nosuid,nodev,size={1 << 30}",
        f"/var/tmp:noexec,nosuid,nodev,size={1 << 30}",
    ]
    env = policy_environment(closed)
    assert env == {
        "HOME": "/scratch/home",
        "TMPDIR": "/scratch",
        "XDG_CACHE_HOME": "/scratch/home/.cache",
        "TRITON_CACHE_DIR": "/scratch/home/.cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/scratch/home/.cache/torchinductor",
        "TORCH_EXTENSIONS_DIR": "/scratch/home/.cache/torch_extensions",
    }
    assert {name for name, _ in CACHE_ENV} <= set(env)
    assert [f"{k}={v}" for k, v in env.items()] == [
        args[i + 1] for i, a in enumerate(args) if a == "--env" and "=" in args[i + 1]
    ]


def test_the_container_command_makes_the_home_then_becomes_the_server(spec, tmp_path):
    """The tmpfs is empty at every start, so a shell makes the home and cache root first and then
    execs the server with its arguments intact (run here on the host, in the same shell syntax)."""
    command = container_argv(spec)
    assert command[:4] == ["sh", "-c", PREPARE_HOME, "sh"] and command[4:] == serve_argv(spec)
    home = tmp_path / "home"
    probe = "import os, sys; print(os.getpid()); print(sys.argv[1:])"
    done = subprocess.Popen(
        [*command[:4], sys.executable, "-c", probe, "an arg with spaces", "$HOME"],
        env={"PATH": os.environ["PATH"], "HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache")},
        stdout=subprocess.PIPE,
        text=True,
    )
    out, _ = done.communicate(timeout=60)
    assert done.returncode == 0
    pid, argv = out.splitlines()
    assert int(pid) == done.pid, "exec: the server is the shell's process, not its child"
    assert argv == "['an arg with spaces', '$HOME']", "arguments pass through unexpanded"
    assert (home / ".cache").is_dir()
    assert (home.stat().st_mode & 0o777, (home / ".cache").stat().st_mode & 0o777) == (0o700, 0o700)
    # A home that cannot be made stops the container before the server starts.
    failed = subprocess.run(
        [*command[:4], sys.executable, "-c", "print('ran')"],
        env={"PATH": os.environ["PATH"], "HOME": "/proc/no-home", "XDG_CACHE_HOME": "/proc/x"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert failed.returncode != 0 and "ran" not in failed.stdout


def test_the_shared_directory_is_private_to_the_sandbox_user(sandbox_spec, tmp_path, shared_mounts):
    shared = tmp_path / "shared"
    (shared / "sub").mkdir(parents=True)
    (shared / "policy.sock").write_text("stale")
    assert prepare_socket_dir(shared, sandbox_spec, bounded=False) is False
    info = shared.stat()
    uid, gid = sandbox_user(sandbox_spec)
    assert (info.st_mode & 0o777, info.st_uid, info.st_gid) == (0o700, uid, gid)
    assert not (shared / "policy.sock").exists(), "a stale socket would be connected to"
    assert shared_mounts == [], "unbounded: a plain directory, nothing mounted"


def test_the_shared_directory_is_a_bounded_tmpfs_for_the_containers_lifetime(
    sandbox_spec, docker, base, ref, tmp_path, shared_mounts
):
    """The policy can write to the one directory it shares with the host, so that directory is
    a tmpfs of SHARED_DIR_BYTES owned by the sandbox user, mounted at start and released at
    close, after the container is gone, with the log kept. (The mount itself is recorded here
    and made for real in the container tests.)"""
    root = write_policy_repo(tmp_path / "repo")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    shared = tmp_path / "s"
    uid, gid = sandbox_user(sandbox_spec)
    container = PolicyContainer(
        sandbox_spec,
        docker,
        built.tag,
        name="vector-policy-bounded",
        socket_dir=shared,
        bounded=True,
    )
    assert container.bounded is True and shared_mounts == []
    container.hello(sandbox_spec.budgets["policy_start_seconds"])
    assert shared_mounts == [("mount", shared, uid, gid)]
    assert SHARED_DIR_BYTES >= 8 << 20 and SHARED_DIR_INODES >= 8, "the socket and the log fit"
    container.close()
    assert shared_mounts == [("mount", shared, uid, gid), ("umount", shared)]
    assert docker.removed == ["vector-policy-bounded"]
    assert (shared / "policy.log").is_file(), "the log outlives the tmpfs"
    assert "listening on" in (shared / "policy.log").read_text()
    # By default a container is bounded exactly when this process can mount a tmpfs: root.
    assert (
        PolicyContainer(sandbox_spec, docker, built.tag, name="n", socket_dir=shared).bounded
        is None
    )
    assert can_bound_shared_dir() == (os.geteuid() == 0)


def test_nothing_written_to_the_shared_tmpfs_runs(monkeypatch, tmp_path):
    """The policy writes its socket and log to the shared directory, and nothing else it writes
    there may run: the tmpfs is mounted nosuid, nodev and noexec, which the bind mount into the
    container keeps (the container tests read it from /proc/mounts), so the spec's tmpfs is the
    only place code a policy writes runs from. `mount_shared_dir` is the real one, imported before
    the pure suite stands a recorder in; only the `mount` command is caught."""
    calls: list[list[str]] = []
    monkeypatch.setattr("vector_orchestrator.submissions.container._mount_command", calls.append)
    mount_shared_dir(tmp_path, 1000, 1001)
    ((*command, options, source, target),) = calls
    assert (command, source, target) == (
        ["mount", "-t", "tmpfs", "-o"],
        SHARED_DIR_SOURCE,
        str(tmp_path),
    )
    flags = options.split(",")
    assert {"nosuid", "nodev", "noexec"} <= set(flags), options
    assert {f"size={SHARED_DIR_BYTES}", f"nr_inodes={SHARED_DIR_INODES}"} <= set(flags), options
    assert {"uid=1000", "gid=1001", "mode=0700"} <= set(flags), options


def test_only_a_socket_itself_counts_as_listening(tmp_path):
    """The policy can write to the socket's directory: a plain file, a link to a socket or to
    anything else at the socket's name is not a server listening."""
    assert not is_socket(tmp_path / "policy.sock"), "nothing there yet"
    (tmp_path / "policy.sock").write_text("not a socket")
    assert not is_socket(tmp_path / "policy.sock")
    (tmp_path / "policy.sock").unlink()
    os.symlink("/etc/passwd", tmp_path / "policy.sock")
    assert not is_socket(tmp_path / "policy.sock"), "exists(), which follows, would say yes"
    (tmp_path / "policy.sock").unlink()
    with socket.socket(socket.AF_UNIX) as real:
        real.bind(str(tmp_path / "real.sock"))
        assert is_socket(tmp_path / "real.sock")
        os.symlink(tmp_path / "real.sock", tmp_path / "policy.sock")
        assert not is_socket(tmp_path / "policy.sock"), "a link to a socket is not the socket"


def test_hello_through_the_container_keeps_the_session_and_removal_follows(
    sandbox_spec, docker, base, ref, tmp_path
):
    root = write_policy_repo(tmp_path / "repo")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    shared = tmp_path / "s"
    with PolicyContainer(
        sandbox_spec, docker, built.tag, name="vector-policy-test", socket_dir=shared, gpus=0
    ) as container:
        reply = container.hello(sandbox_spec.budgets["policy_start_seconds"])
        assert reply == {"protocol": 1, "action_type": "qpos", "policy": "pkg.policy:Policy"}
        assert container.session is not None and container.session.action_type == "qpos"
        assert container.listening_after_s is not None and container.listening_after_s < 30
        # The start budget was for hello; what drives the policy next gets the act budget.
        assert container.session.timeout_s == sandbox_spec.budgets["act_timeout_s"]
        assert container.session.timeout_s < sandbox_spec.budgets["policy_start_seconds"]
        assert container.session.act({"obs": [0.0]}) == {"action": [0.0]}
        # The server unlinks the socket once its one client is in; the log stays.
        assert not container.socket_path.exists() and container.log_path.exists()
        assert docker.state("vector-policy-test").running and docker.removed == []
        # The key crossed by variable name, in the docker client's environment, and nowhere on
        # the command line; it is 32 random bytes.
        (env,) = docker.run_envs
        assert set(env) == {AUTHKEY_ENV} and len(bytes.fromhex(env[AUTHKEY_ENV])) == 32
        assert env[AUTHKEY_ENV] == container.authkey.hex()
        assert not any(env[AUTHKEY_ENV] in arg for arg in docker.runs[0])
        assert "--gpus" not in docker.runs[0]
    assert (
        docker.removed == ["vector-policy-test"] and not docker.state("vector-policy-test").running
    )
    container.close()
    assert docker.removed == ["vector-policy-test"], "removed once"


def test_an_owner_is_gone_when_its_process_is_and_not_while_it_runs():
    me = Owner.current()
    assert me is not None and me.pid == os.getpid() and me.pidns.startswith("pid:[")
    assert not me.gone()
    assert Owner.from_labels(me.labels()) == me and Owner.from_labels({}) is None
    ended = ended_process_owner()
    assert ended.gone(), "the process has exited"
    # A later process given the same pid started later: the owner is still gone.
    assert Owner(me.pid, str(int(me.start) + 1), me.pidns).gone()


def test_start_reaps_the_containers_of_processes_that_are_gone_and_nothing_else(
    sandbox_spec, docker, base, ref, tmp_path
):
    """A process killed between `docker run` and `hello` leaves a server waiting for a client
    for ever. Each start removes the containers whose owner has ended; one whose owner lives,
    one from another pid namespace and one with no owner labels are not this process's to judge."""
    ended = ended_process_owner()
    elsewhere = Owner(ended.pid, ended.start, "pid:[1]")
    policy = {"vector.orchestrator": "policy"}
    docker.labels["vector-policy-orphan"] = {
        **policy,
        **ended.labels(),
        "vector.shared-dir": str(tmp_path / "gone"),
    }
    docker.labels["vector-policy-alive"] = {**policy, **Owner.current().labels()}
    docker.labels["vector-policy-elsewhere"] = {**policy, **elsewhere.labels()}
    docker.labels["vector-policy-unlabelled"] = dict(policy)
    root = write_policy_repo(tmp_path / "repo")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    with PolicyContainer(
        sandbox_spec, docker, built.tag, name="vector-policy-new", socket_dir=tmp_path / "s", gpus=0
    ) as container:
        assert container.reaped == ["vector-policy-orphan"]
        assert docker.removed == ["vector-policy-orphan"]
        # The new container names its owner - this process - and its shared directory.
        labels = docker.labels["vector-policy-new"]
        assert Owner.from_labels(labels) == Owner.current()
        assert labels["vector.shared-dir"] == str((tmp_path / "s").absolute())
    assert sorted(docker.labels) == [
        "vector-policy-alive",
        "vector-policy-elsewhere",
        "vector-policy-unlabelled",
    ]
    # Where /proc cannot say who this process is, or docker cannot list, nothing is reaped.
    docker.labels["vector-policy-orphan"] = {**policy, **ended.labels()}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(docker, "policy_containers", lambda: raise_(DockerError("refused")))
        assert reap_orphans(docker) == []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Owner, "current", classmethod(lambda cls: None))
        assert reap_orphans(docker) == []
    assert "vector-policy-orphan" in docker.labels
    assert reap_orphans(docker) == ["vector-policy-orphan"], "and once it can, it is"


def raise_(exc: Exception):
    raise exc


def test_a_manifest_naming_a_missing_class_is_rejected_at_hello_and_the_container_removed(
    sandbox_spec, docker, base, ref, tmp_path
):
    root = write_policy_repo(tmp_path / "repo", policy="pkg.policy:Missing")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    container = PolicyContainer(
        sandbox_spec, docker, built.tag, name="vector-policy-missing", socket_dir=tmp_path / "s"
    )
    with pytest.raises(SubmissionRejected) as info:
        container.hello(sandbox_spec.budgets["policy_start_seconds"])
    assert info.value.step == "hello"
    assert "module 'pkg.policy' has no attribute 'Missing'" in info.value.reason
    assert "--- policy log (tail) ---" in info.value.reason, "the server's log travels with it"
    assert container.session is None
    container.close()
    assert docker.removed == ["vector-policy-missing"]


def test_a_container_that_exits_before_listening_is_rejected_at_start(
    sandbox_spec, docker, base, ref, tmp_path
):
    root = write_policy_repo(tmp_path / "repo")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    (root / "policy.yaml").unlink()  # the server has nothing to serve and exits 2 at once
    with PolicyContainer(
        sandbox_spec, docker, built.tag, name="vector-policy-dead", socket_dir=tmp_path / "s"
    ) as container:
        with pytest.raises(SubmissionRejected) as info:
            container.hello(sandbox_spec.budgets["policy_start_seconds"])
    assert info.value.step == "start"
    assert "exited (2) before listening" in info.value.reason
    assert docker.removed == ["vector-policy-dead"]


def test_a_policy_that_takes_longer_than_the_budget_to_build_is_rejected(
    sandbox_spec, docker, base, ref, tmp_path
):
    root = write_policy_repo(tmp_path / "repo")
    (root / "pkg" / "policy.py").write_text(
        "import time\n"
        "class Policy:\n"
        "    action_type = 'qpos'\n"
        "    def __init__(self): time.sleep(30)\n"
        "    def reset(self, seed): pass\n"
        "    def set_demonstration(self, arrays, info): pass\n"
        "    def act(self, observation): pass\n"
    )
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    started = time.monotonic()
    with PolicyContainer(
        sandbox_spec, docker, built.tag, name="vector-policy-slow", socket_dir=tmp_path / "s"
    ) as container:
        with pytest.raises(SubmissionRejected) as info:
            container.hello(3.0)
    assert info.value.step == "hello" and "no answer within" in info.value.reason
    assert time.monotonic() - started < 20
