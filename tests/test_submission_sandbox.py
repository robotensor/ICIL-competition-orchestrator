"""A submission's container, with Docker stood in for.

`FakeDocker.run` starts the real `python -m icil_policy.serve` on the host in the container's
place, so what is tested is the orchestrator's side of the sandbox: the arguments it would give
`docker run`, the socket directory, the authkey, the wait for `hello` and the removal - not the
kernel's isolation, which the container tests check for real.
"""

from __future__ import annotations

import time

import pytest

from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.submissions import SubmissionRejected
from icil_orchestrator.submissions.checks import check_repository
from icil_orchestrator.submissions.container import (
    AUTHKEY_ENV,
    HARDENING,
    PolicyContainer,
    prepare_socket_dir,
    run_argv,
    serve_argv,
)
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


# -- the container ------------------------------------------------------------------------------


def test_run_argv_is_exactly_the_specs_sandbox_and_one_shared_directory(spec, tmp_path):
    sandbox = spec.submission["sandbox"]
    args = run_argv(spec, image="icil-submission:k-s", name="icil-policy-x", socket_dir=tmp_path)
    assert args == [
        "--detach",
        "--name",
        "icil-policy-x",
        "--label",
        "icil.orchestrator=policy",
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


def test_the_shared_directory_is_private_to_the_sandbox_user(sandbox_spec, tmp_path):
    shared = tmp_path / "shared"
    (shared / "sub").mkdir(parents=True)
    (shared / "policy.sock").write_text("stale")
    prepare_socket_dir(shared, sandbox_spec)
    info = shared.stat()
    uid, gid = sandbox_user(sandbox_spec)
    assert (info.st_mode & 0o777, info.st_uid, info.st_gid) == (0o700, uid, gid)
    assert not (shared / "policy.sock").exists(), "a stale socket would be connected to"


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
