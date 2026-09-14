"""The policy runtime seam, through the subprocess runtime the pure suite duels with."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from duel_helpers import REPLAY, REPLAY_REF, ZERO, FakePolicyRuntime
from icil_orchestrator.duel.local_runtime import SubprocessPolicyRuntime, tree_hash
from icil_orchestrator.duel.runtime import (
    POLICY,
    PolicyDied,
    PolicyRuntime,
    RuntimeUnavailable,
    SubmissionRefused,
)
from submission_helpers import write_policy_repo


@pytest.fixture
def runtime(spec):
    return SubprocessPolicyRuntime(
        spec, {"org/replay": REPLAY, "org/zero": ZERO}, start_timeout_s=30
    )


def test_the_subprocess_runtime_is_a_policy_runtime(runtime):
    assert isinstance(runtime, PolicyRuntime) and runtime.name == "local"


def test_resolve_keeps_a_commit_and_pins_a_name_to_the_directory(runtime):
    assert runtime.resolve("org/replay", "a" * 40).revision == "a" * 40
    pinned = runtime.resolve("org/replay", "main")
    assert pinned.revision == tree_hash(REPLAY) and pinned.repo == "org/replay"
    with pytest.raises(RuntimeUnavailable, match="no local directory for org/other"):
        runtime.resolve("org/other", "main")


def test_prepare_checks_the_manifest_and_says_hello(runtime, tmp_path):
    ref = runtime.resolve("org/replay", "a" * 40)
    fetched = runtime.fetch(ref, workdir=tmp_path)
    assert fetched.commit == "a" * 40 and fetched.root == REPLAY.resolve()
    prepared = runtime.prepare(fetched, workdir=tmp_path / "check")
    assert prepared.policy == "replay.policy:ReplayPolicy" and prepared.action_type == "qpos"
    assert prepared.as_side() == {
        "commit": "a" * 40,
        "base_image_digest": None,
        "image": None,
        "policy": "replay.policy:ReplayPolicy",
        "action_type": "qpos",
    }
    assert "listening" in (tmp_path / "check" / "policy.log").read_text()


def test_a_manifest_naming_a_missing_class_is_refused_at_hello(spec, tmp_path):
    repo = write_policy_repo(tmp_path / "repo", policy="pkg.policy:Missing")
    runtime = SubprocessPolicyRuntime(spec, {"org/broken": repo}, start_timeout_s=30)
    fetched = runtime.fetch(runtime.resolve("org/broken", "b" * 40), workdir=tmp_path)
    with pytest.raises(SubmissionRefused) as refused:
        runtime.prepare(fetched, workdir=tmp_path / "check")
    assert refused.value.step == "hello" and "Missing" in refused.value.reason


def test_a_directory_without_a_manifest_is_refused(spec, tmp_path):
    (tmp_path / "empty").mkdir()
    runtime = SubprocessPolicyRuntime(spec, {"org/empty": tmp_path / "empty"})
    fetched = runtime.fetch(runtime.resolve("org/empty", "c" * 40), workdir=tmp_path)
    with pytest.raises(SubmissionRefused) as refused:
        runtime.prepare(fetched, workdir=tmp_path)
    assert refused.value.step == "manifest"


def test_each_serve_is_a_fresh_server_with_its_own_key_gone_after_its_unit(runtime, tmp_path):
    from icil_policy.client import RemotePolicy

    prepared = runtime.prepare(
        runtime.fetch(runtime.resolve("org/zero", "d" * 40), workdir=tmp_path), workdir=tmp_path
    )
    seen = []
    for unit in ("fp-000", "fs-001"):
        with runtime.serve(prepared, workdir=tmp_path / unit) as served:
            key = bytes.fromhex(served.env[served.authkey_env])
            assert list(served.env) == [served.authkey_env] and len(key) == 32
            with RemotePolicy(served.address, key, timeout_s=10) as policy:
                assert policy.hello()["policy"] == "zero.policy:ZeroPolicy"
            assert served.died() is None, "a server that ended with its client did not die"
            seen.append((served.address, key))
        assert not Path(served.address).parent.exists(), "the socket directory outlived its unit"
        assert served.log_file == tmp_path / unit / "policy.log" and served.log_file.is_file()
    assert seen[0][0] != seen[1][0] and seen[0][1] != seen[1][1]


def test_a_server_killed_under_its_unit_is_reported_dead_by_its_own_doing(spec, tmp_path):
    runtime = FakePolicyRuntime(spec, kill_on_serve={0})
    prepared = runtime.prepare(runtime.fetch(REPLAY_REF, workdir=tmp_path), workdir=tmp_path)
    with runtime.serve(prepared, workdir=tmp_path / "fp-000") as served:
        end = served.died()
        assert end.cause == POLICY
        assert end.reason.startswith("the policy process was killed by signal 9")


def test_a_server_that_never_listens_is_a_dead_policy(runtime, tmp_path):
    prepared = runtime.prepare(
        runtime.fetch(runtime.resolve("org/zero", "e" * 40), workdir=tmp_path), workdir=tmp_path
    )
    runtime.python = "/bin/false"
    started = time.monotonic()
    with pytest.raises(PolicyDied, match=r"exited \(1\) before listening"):
        with runtime.serve(prepared, workdir=tmp_path / "u"):
            pass
    assert time.monotonic() - started < 10
