"""`icil-orchestrator submission check|build-base`, with Docker stood in for.

The fake starts the real policy server on the host in the container's place, so `check` on the
replay example goes through every step - resolve, fetch, manifest, build, start, hello - and the
report is the one a duel would carry.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from icil_orchestrator.cli import main
from icil_orchestrator.ids import is_commit_sha
from icil_orchestrator.submissions.check import BUILD_TIMEOUT_S, STEPS, check_submission
from icil_orchestrator.submissions.fetch import LocalFetcher, RepoCache
from submission_helpers import FAKE_BASE_DIGEST, FakeDocker

EXAMPLE = Path(__file__).resolve().parents[1] / "packages/icil-policy/examples/replay_policy"


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    fake.images["icil-policy-base:latest"] = FAKE_BASE_DIGEST
    monkeypatch.setattr("icil_orchestrator.submissions.docker.Docker", lambda *a, **k: fake)
    yield fake
    fake.kill_all()


def check(sandbox_spec, tmp_path, *args):
    return main(
        [
            "--spec",
            str(sandbox_spec.path),
            "submission",
            "check",
            *args,
            "--cache",
            str(tmp_path / "cache"),
            "--work",
            str(tmp_path / "work"),
            "--gpus",
            "0",
        ]
    )


def test_check_takes_the_replay_example_through_every_step_and_reports_them(
    sandbox_spec, fake_docker, tmp_path, capsys
):
    code = check(
        sandbox_spec,
        tmp_path,
        "local/replay_policy@main",
        "--local",
        str(EXAMPLE),
        "--base-image",
        FAKE_BASE_DIGEST,
    )
    out = capsys.readouterr().out
    assert code == 0, out
    lines = out.strip().splitlines()
    assert [line.split()[:2] for line in lines[:-1]] == [[name, "ok"] for name in STEPS]
    assert lines[-1].startswith("local/replay_policy@") and lines[-1].endswith(": ACCEPTED")
    sha = lines[-1].split("@", 1)[1].split(":")[0]
    assert is_commit_sha(sha)
    assert "policy replay.policy:ReplayPolicy, requirements requirements.txt" in lines[2]
    assert f"FROM icil-policy-base {FAKE_BASE_DIGEST}" in lines[3]
    assert "action_type qpos" in lines[5] and "listening after" in lines[5]
    assert fake_docker.removed == [fake_docker.runs[0][fake_docker.runs[0].index("--name") + 1]]
    assert (tmp_path / "work" / "policy" / "policy.log").is_file(), "--work keeps the log"
    assert fake_docker.build_timeouts == [BUILD_TIMEOUT_S], "the build is bounded by default"

    # Again, as JSON: the checkout is cached, the image is built again (Docker's cache is its
    # own), and the side an event would record names the commit and the base image.
    code = check(
        sandbox_spec,
        tmp_path,
        "local/replay_policy@main",
        "--local",
        str(EXAMPLE),
        "--base-image",
        FAKE_BASE_DIGEST,
        "--json",
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["verdict"] == "accepted" and report["sha"] == sha
    assert "(cached)" in report["steps"][1]["detail"]
    assert report["side"] == {
        "key": report["key"],
        "repo": "local/replay_policy",
        "revision": sha,
        "base_image": FAKE_BASE_DIGEST,
        "image": report["image"],
        "policy": "replay.policy:ReplayPolicy",
        "action_type": "qpos",
        "protocol": 1,
        "start_seconds": report["listening_after_s"],
        "verdict": "accepted",
        "reason": None,
    }
    assert report["image"].startswith("sha256:") and report["build_seconds"] >= 0


def test_check_without_a_base_image_digest_is_an_error_at_build_not_a_verdict(
    sandbox_spec, fake_docker, tmp_path, capsys
):
    assert sandbox_spec.submission["base_image"]["digest"] is None
    code = check(sandbox_spec, tmp_path, "local/replay_policy@main", "--local", str(EXAMPLE))
    out = capsys.readouterr().out
    assert code == 2 and out.strip().endswith(": ERROR at build"), out
    assert "no base image digest" in out and "submission build-base" in out
    assert fake_docker.runs == []


def test_check_rejects_requirements_that_do_not_install_and_runs_nothing(
    sandbox_spec, fake_docker, tmp_path, capsys
):
    broken = shutil.copytree(EXAMPLE, tmp_path / "broken")
    (broken / "requirements.txt").write_text("icil-no-such-package==99.0\n")
    fake_docker.build_failure = "ERROR: No matching distribution found for icil-no-such-package"
    code = check(
        sandbox_spec,
        tmp_path,
        "local/broken@main",
        "--local",
        str(broken),
        "--base-image",
        FAKE_BASE_DIGEST,
    )
    out = capsys.readouterr().out
    assert code == 1 and out.strip().endswith(": REJECTED at build"), out
    assert "installing requirements.txt failed" in out and "No matching distribution" in out
    statuses = [line.split()[1] for line in out.splitlines() if line.split()[:1][0] in STEPS]
    assert statuses == ["ok", "ok", "ok", "rejected", "skipped", "skipped"]
    assert fake_docker.runs == [] and fake_docker.removed == []


def test_check_rejects_requirements_that_never_finish_installing_at_build(
    sandbox_spec, fake_docker, tmp_path, capsys
):
    fake_docker.build_seconds = 10.0
    code = check(
        sandbox_spec,
        tmp_path,
        "local/replay_policy@main",
        "--local",
        str(EXAMPLE),
        "--base-image",
        FAKE_BASE_DIGEST,
        "--build-timeout",
        "5",
    )
    out = capsys.readouterr().out
    assert code == 1 and out.strip().endswith(": REJECTED at build"), out
    assert "installing requirements.txt did not finish within 5s" in out
    assert fake_docker.build_timeouts == [5.0] and fake_docker.runs == []


def test_check_rejects_a_manifest_naming_a_missing_class_at_hello(
    sandbox_spec, fake_docker, tmp_path
):
    broken = shutil.copytree(EXAMPLE, tmp_path / "broken")
    (broken / "icil.yaml").write_text("api: 1\npolicy: replay.policy:NoSuchPolicy\n")
    cache = RepoCache(tmp_path / "cache", sandbox_spec.submission["max_repo_bytes"])
    report = check_submission(
        sandbox_spec,
        "local/broken",
        "main",
        fetcher=LocalFetcher(cache, broken),
        docker=fake_docker,
        work_dir=tmp_path / "work",
        base_digest=FAKE_BASE_DIGEST,
        gpus=0,
    )
    assert report.verdict == "rejected" and report.failed_step.name == "hello"
    assert "has no attribute 'NoSuchPolicy'" in report.failed_step.detail
    assert [s.status for s in report.steps] == ["ok", "ok", "ok", "ok", "ok", "rejected"]
    assert report.side()["revision"] == report.sha and report.side()["base_image"] == (
        FAKE_BASE_DIGEST
    )
    assert report.side()["verdict"] == "rejected" and "NoSuchPolicy" in report.side()["reason"]
    assert len(fake_docker.removed) == 1, "the container is gone"


def test_check_refuses_a_ref_that_is_not_repo_at_revision(
    sandbox_spec, fake_docker, tmp_path, capsys
):
    assert check(sandbox_spec, tmp_path, "local/replay_policy", "--local", str(EXAMPLE)) == 2
    # A local directory stands in for the Hub, not for the shape of a ref: what is not a repo id
    # is rejected at resolve, as the Hub path rejects it, not a traceback.
    code = check(sandbox_spec, tmp_path, "not a repo id@main", "--local", str(EXAMPLE))
    out = capsys.readouterr().out
    assert code == 1 and out.strip().endswith(": REJECTED at resolve"), out
    assert "'not a repo id' is not a Hugging Face repo id" in out
    assert fake_docker.builds == [] and fake_docker.runs == []


def test_build_base_prints_the_digest(spec, fake_docker, capsys):
    assert main(["submission", "build-base", "--context", str(EXAMPLE.parents[3])]) == 0
    out = capsys.readouterr()
    digest = out.out.strip()
    assert digest.startswith("sha256:") and len(digest) == 71
    assert f"built icil-policy-base:{digest[7:]} in" in out.err
    assert fake_docker.image_id(f"icil-policy-base:{digest[7:]}") == digest
    assert main(["submission", "build-base", "--context", str(EXAMPLE.parents[3]), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["base"]["digest"] == digest
