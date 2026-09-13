"""A submission's container, with Docker stood in for.

`FakeDocker.run` starts the real `python -m icil_policy.serve` on the host in the container's
place, so what is tested is the orchestrator's side of the sandbox: the arguments it would give
`docker run`, the socket directory, the authkey, the wait for `hello` and the removal - not the
kernel's isolation, which the container tests check for real.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.submissions import SubmissionRejected
from icil_orchestrator.submissions.checks import check_repository
from icil_orchestrator.submissions.container import (
    AUTHKEY_ENV,
    HARDENING,
    SHARED_DIR_BYTES,
    SHARED_DIR_INODES,
    Owner,
    PolicyContainer,
    can_bound_shared_dir,
    is_socket,
    prepare_socket_dir,
    reap_orphans,
    run_argv,
    serve_argv,
)
from icil_orchestrator.submissions.docker import DockerError
from icil_orchestrator.submissions.image import (
    base_image,
    build_submission_image,
    sandbox_user,
)
from submission_helpers import FAKE_BASE_DIGEST, SHA_A, FakeDocker, write_policy_repo


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
        spec, image="icil-submission:k-s", name="icil-policy-x", socket_dir=tmp_path, owner=owner
    )
    assert args == [
        "--detach",
        "--name",
        "icil-policy-x",
        "--label",
        "icil.orchestrator=policy",
        "--label",
        "icil.owner.pid=4321",
        "--label",
        "icil.owner.start=98765",
        "--label",
        "icil.owner.pidns=pid:[4026531836]",
        "--label",
        f"icil.shared-dir={tmp_path}",
        "--network",
        sandbox["network"],
        "--read-only",
        *[flag for path in sandbox["tmpfs"] for flag in ("--tmpfs", path)],
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
        f"type=bind,src={tmp_path},dst=/run/icil",
        "--env",
        AUTHKEY_ENV,
        "icil-submission:k-s",
        *serve_argv(spec),
    ]
    assert sandbox["network"] == "none" and sandbox["read_only_root"] is True
    assert args.count("--mount") == 1 and "--volume" not in args and "-v" not in args
    assert serve_argv(spec) == [
        "python",
        "-m",
        "icil_policy.serve",
        "--manifest",
        f"/submission/{spec.submission['manifest']}",
        "--address",
        "/run/icil/policy.sock",
        "--authkey-env",
        AUTHKEY_ENV,
        "--log-file",
        "/run/icil/policy.log",
    ]
    # A policy that needs no GPU gets none; the spec's count is the default.
    without = run_argv(spec, image="i", name="n", socket_dir=tmp_path, gpus=0)
    assert "--gpus" not in without and without.count("--memory") == 1


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
        sandbox_spec, docker, built.tag, name="icil-policy-bounded", socket_dir=shared, bounded=True
    )
    assert container.bounded is True and shared_mounts == []
    container.hello(sandbox_spec.budgets["policy_start_seconds"])
    assert shared_mounts == [("mount", shared, uid, gid)]
    assert SHARED_DIR_BYTES >= 8 << 20 and SHARED_DIR_INODES >= 8, "the socket and the log fit"
    container.close()
    assert shared_mounts == [("mount", shared, uid, gid), ("umount", shared)]
    assert docker.removed == ["icil-policy-bounded"]
    assert (shared / "policy.log").is_file(), "the log outlives the tmpfs"
    assert "listening on" in (shared / "policy.log").read_text()
    # By default a container is bounded exactly when this process can mount a tmpfs: root.
    assert (
        PolicyContainer(sandbox_spec, docker, built.tag, name="n", socket_dir=shared).bounded
        is None
    )
    assert can_bound_shared_dir() == (os.geteuid() == 0)


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
        sandbox_spec, docker, built.tag, name="icil-policy-test", socket_dir=shared, gpus=0
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
        assert docker.state("icil-policy-test").running and docker.removed == []
        # The key crossed by variable name, in the docker client's environment, and nowhere on
        # the command line; it is 32 random bytes.
        (env,) = docker.run_envs
        assert set(env) == {AUTHKEY_ENV} and len(bytes.fromhex(env[AUTHKEY_ENV])) == 32
        assert env[AUTHKEY_ENV] == container.authkey.hex()
        assert not any(env[AUTHKEY_ENV] in arg for arg in docker.runs[0])
        assert "--gpus" not in docker.runs[0]
    assert docker.removed == ["icil-policy-test"] and not docker.state("icil-policy-test").running
    container.close()
    assert docker.removed == ["icil-policy-test"], "removed once"


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
    policy = {"icil.orchestrator": "policy"}
    docker.labels["icil-policy-orphan"] = {
        **policy,
        **ended.labels(),
        "icil.shared-dir": str(tmp_path / "gone"),
    }
    docker.labels["icil-policy-alive"] = {**policy, **Owner.current().labels()}
    docker.labels["icil-policy-elsewhere"] = {**policy, **elsewhere.labels()}
    docker.labels["icil-policy-unlabelled"] = dict(policy)
    root = write_policy_repo(tmp_path / "repo")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    with PolicyContainer(
        sandbox_spec, docker, built.tag, name="icil-policy-new", socket_dir=tmp_path / "s", gpus=0
    ) as container:
        assert container.reaped == ["icil-policy-orphan"]
        assert docker.removed == ["icil-policy-orphan"]
        # The new container names its owner - this process - and its shared directory.
        labels = docker.labels["icil-policy-new"]
        assert Owner.from_labels(labels) == Owner.current()
        assert labels["icil.shared-dir"] == str((tmp_path / "s").absolute())
    assert sorted(docker.labels) == [
        "icil-policy-alive",
        "icil-policy-elsewhere",
        "icil-policy-unlabelled",
    ]
    # Where /proc cannot say who this process is, or docker cannot list, nothing is reaped.
    docker.labels["icil-policy-orphan"] = {**policy, **ended.labels()}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(docker, "policy_containers", lambda: raise_(DockerError("refused")))
        assert reap_orphans(docker) == []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Owner, "current", classmethod(lambda cls: None))
        assert reap_orphans(docker) == []
    assert "icil-policy-orphan" in docker.labels
    assert reap_orphans(docker) == ["icil-policy-orphan"], "and once it can, it is"


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
        sandbox_spec, docker, built.tag, name="icil-policy-missing", socket_dir=tmp_path / "s"
    )
    with pytest.raises(SubmissionRejected) as info:
        container.hello(sandbox_spec.budgets["policy_start_seconds"])
    assert info.value.step == "hello"
    assert "module 'pkg.policy' has no attribute 'Missing'" in info.value.reason
    assert "--- policy log (tail) ---" in info.value.reason, "the server's log travels with it"
    assert container.session is None
    container.close()
    assert docker.removed == ["icil-policy-missing"]


def test_a_container_that_exits_before_listening_is_rejected_at_start(
    sandbox_spec, docker, base, ref, tmp_path
):
    root = write_policy_repo(tmp_path / "repo")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(
        docker, sandbox_spec, root, check_repository(root, sandbox_spec), ref, base
    )
    (root / "icil.yaml").unlink()  # the server has nothing to serve and exits 2 at once
    with PolicyContainer(
        sandbox_spec, docker, built.tag, name="icil-policy-dead", socket_dir=tmp_path / "s"
    ) as container:
        with pytest.raises(SubmissionRejected) as info:
            container.hello(sandbox_spec.budgets["policy_start_seconds"])
    assert info.value.step == "start"
    assert "exited (2) before listening" in info.value.reason
    assert docker.removed == ["icil-policy-dead"]


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
        sandbox_spec, docker, built.tag, name="icil-policy-slow", socket_dir=tmp_path / "s"
    ) as container:
        with pytest.raises(SubmissionRejected) as info:
            container.hello(3.0)
    assert info.value.step == "hello" and "no answer within" in info.value.reason
    assert time.monotonic() - started < 20
