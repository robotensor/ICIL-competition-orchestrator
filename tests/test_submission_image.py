"""The pinned base image and a submission's image built from it, with Docker stood in for."""

from __future__ import annotations

from pathlib import Path

import pytest

from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.submissions import SubmissionError, SubmissionRejected
from icil_orchestrator.submissions.checks import check_repository
from icil_orchestrator.submissions.image import (
    BASE_DOCKERFILE,
    INDEX_PROBE_IMAGE,
    INDEX_PROBE_REQUIREMENT,
    INDEX_PROBE_TAIL_CHARS,
    INDEX_PROBE_TIMEOUT_S,
    BaseImage,
    base_image,
    build_base_image,
    build_submission_image,
    ensure_base,
    index_probe_dockerfile,
    probe_index,
    sandbox_user,
    submission_dockerfile,
    submission_tag,
)
from submission_helpers import (
    FAKE_BASE_DIGEST,
    SHA_A,
    FakeDocker,
    pip_unreachable_log,
    write_policy_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def docker():
    return FakeDocker()


@pytest.fixture
def base(spec):
    return base_image(spec, FAKE_BASE_DIGEST)


@pytest.fixture
def ref():
    return SubmissionRef.resolved("org/policy", SHA_A)


# -- the base image -----------------------------------------------------------------------------


def test_the_base_is_the_specs_name_at_the_digest_pinned_or_given(spec):
    assert spec.submission["base_image"]["digest"] is None, "the spec's pin is still open"
    with pytest.raises(SubmissionError, match="no base image digest.*submission build-base"):
        base_image(spec)
    base = base_image(spec, FAKE_BASE_DIGEST)
    assert base == BaseImage(name=spec.submission["base_image"]["name"], digest=FAKE_BASE_DIGEST)
    assert base.tag == f"{spec.submission['base_image']['name']}:{'b' * 64}"
    with pytest.raises(SubmissionError, match="is not an image digest"):
        base_image(spec, "icil-policy-base:latest")


def test_build_base_image_builds_the_dockerfile_from_the_repository_root(spec, docker):
    built = build_base_image(docker, spec, REPO_ROOT)
    (context, dockerfile, tag, args), *rest = docker.builds
    assert not rest and context == REPO_ROOT
    assert dockerfile == (REPO_ROOT / BASE_DOCKERFILE).read_text()
    uid, gid = sandbox_user(spec)
    assert args == {"POLICY_UID": str(uid), "POLICY_GID": str(gid)}
    assert tag == f"{built.base.name}:latest"
    assert built.base.digest == built.image_id and built.image_id.startswith("sha256:")
    assert docker.image_id(built.base.tag) == built.image_id, "tagged by its own digest"


def test_ensure_base_wants_exactly_the_digest_on_this_host(spec, docker, base):
    with pytest.raises(SubmissionError, match="is not on this host"):
        ensure_base(docker, base)
    docker.images["icil-policy-base:latest"] = "sha256:" + "c" * 64
    with pytest.raises(SubmissionError, match="is not on this host"):
        ensure_base(docker, base), "another build under the name is not the pinned base"
    docker.images["something:else"] = FAKE_BASE_DIGEST
    assert ensure_base(docker, base) == base.tag
    assert docker.images[base.tag] == FAKE_BASE_DIGEST and docker.tags == [
        (FAKE_BASE_DIGEST, base.tag)
    ]
    assert ensure_base(docker, base) == base.tag and len(docker.tags) == 1


# -- a submission's image -----------------------------------------------------------------------


def test_the_submissions_dockerfile_is_the_base_the_checkout_and_its_requirements(
    spec, base, tmp_path
):
    root = write_policy_repo(tmp_path / "repo", requirements="requirements.txt")
    (root / "requirements.txt").write_text("numpy\n")
    uid, gid = sandbox_user(spec)
    assert submission_dockerfile(base, check_repository(root, spec), spec) == (
        f"# {FAKE_BASE_DIGEST}\n"
        f"FROM {base.tag}\n"
        f"COPY --chown={uid}:{gid} . /submission\n"
        "RUN python -m pip install --no-cache-dir -r /submission/requirements.txt\n"
    )
    bare = write_policy_repo(tmp_path / "bare")
    assert submission_dockerfile(base, check_repository(bare, spec), spec) == (
        f"# {FAKE_BASE_DIGEST}\nFROM {base.tag}\nCOPY --chown={uid}:{gid} . /submission\n"
    )


def test_build_submission_image_tags_by_key_and_sha_from_the_verified_base(
    spec, docker, base, ref, tmp_path
):
    root = write_policy_repo(tmp_path / "repo")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    built = build_submission_image(docker, spec, root, check_repository(root, spec), ref, base)
    assert built.tag == submission_tag(ref) == f"icil-submission:{ref.key}-{SHA_A}"
    assert built.base == base and docker.image_id(built.tag) == built.image_id
    context, dockerfile, tag, _ = docker.builds[-1]
    assert (context, tag) == (root, built.tag) and dockerfile.startswith(f"# {FAKE_BASE_DIGEST}")


def test_requirements_that_do_not_install_reject_the_submission_with_the_reason(
    spec, docker, base, ref, tmp_path
):
    root = write_policy_repo(tmp_path / "repo", requirements="requirements.txt")
    (root / "requirements.txt").write_text("icil-no-such-package==99.0\n")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    docker.build_failure = "ERROR: No matching distribution found for icil-no-such-package==99.0"
    with pytest.raises(SubmissionRejected) as info:
        build_submission_image(docker, spec, root, check_repository(root, spec), ref, base)
    assert info.value.step == "build"
    assert info.value.reason.startswith("installing requirements.txt failed:")
    assert "No matching distribution found for icil-no-such-package" in info.value.reason
    assert docker.runs == [], "nothing ran"
    assert len(docker.probes) == 1, "the orchestrator's own probe reached the index first"


def test_the_index_probe_is_pip_in_the_base_with_nothing_of_any_submission(spec, docker, base):
    with pytest.raises(SubmissionError, match="is not on this host"):
        probe_index(docker, base)
    docker.images["x:y"] = FAKE_BASE_DIGEST
    probe = probe_index(docker, base)
    assert probe.reachable and probe.detail.startswith("pip download pip worked in")
    (seen,) = docker.probes
    assert seen.dockerfile == index_probe_dockerfile(base)
    lines = seen.dockerfile.splitlines()
    assert lines[1] == f"FROM {base.tag}" and len(lines) == 3
    assert lines[2].startswith("RUN python -m pip download --no-deps ")
    assert f" {INDEX_PROBE_REQUIREMENT} && rm -rf " in lines[2]
    assert not any(line.split()[0] in ("COPY", "ADD") for line in lines[1:]), "nothing copied in"
    assert seen.contents == [] and not seen.context.exists(), "an empty context, gone after"
    assert seen.no_cache, "a cached step would pass with no network"
    assert seen.timeout_s == INDEX_PROBE_TIMEOUT_S
    assert seen.tag.startswith(f"{INDEX_PROBE_IMAGE}:") and seen.tag not in docker.images
    probe_index(docker, base)
    assert docker.probes[1].tag != seen.tag, "two probes never share a tag"

    docker.index_reachable = False
    probe = probe_index(docker, base)
    assert not probe.reachable and "Temporary failure in name resolution" in probe.detail
    assert len(probe.detail) <= INDEX_PROBE_TAIL_CHARS


def test_a_build_that_prints_pips_network_failure_is_rejected_when_the_index_answers(
    spec, docker, base, ref, tmp_path
):
    """The build's log is the submission's to write: a `setup.py` that prints pip's words for a
    broken connection and fails is a rejection, because the orchestrator's own probe, which the
    submission has no part in, reached the index."""
    root = write_policy_repo(tmp_path / "repo", requirements="requirements.txt")
    (root / "requirements.txt").write_text("./spoof\n")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    docker.build_failure = pip_unreachable_log("numpy")
    with pytest.raises(SubmissionRejected) as info:
        build_submission_image(docker, spec, root, check_repository(root, spec), ref, base)
    assert info.value.step == "build"
    assert info.value.reason.startswith("installing requirements.txt failed:")
    assert "Temporary failure in name resolution" in info.value.reason
    (probe,) = docker.probes
    assert probe.context != root and probe.contents == []
    assert docker.runs == []


@pytest.mark.parametrize("how", ["fails", "runs out of time"])
def test_a_build_is_the_harness_error_when_the_probe_cannot_reach_the_index_either(
    how, spec, docker, base, ref, tmp_path
):
    """Whatever the build printed - here only the final ERROR lines, which a package that does
    not exist ends in too - with the index out of the probe's reach nobody can tell whose failure
    it was, and it is not published."""
    root = write_policy_repo(tmp_path / "repo", requirements="requirements.txt")
    (root / "requirements.txt").write_text("numpy\n")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    if how == "fails":
        docker.build_failure = "ERROR: No matching distribution found for numpy"
    else:
        docker.build_seconds = 100.0
    docker.index_reachable = False
    with pytest.raises(
        SubmissionError, match="own probe could not reach the index either.*try again later"
    ) as e:
        build_submission_image(
            docker, spec, root, check_repository(root, spec), ref, base, timeout_s=25.0
        )
    assert not isinstance(e.value, SubmissionRejected)
    assert "Temporary failure in name resolution" in str(e.value), "the probe's own words"
    if how == "fails":
        assert "No matching distribution found for numpy" in str(e.value)
    else:
        assert "did not finish within 25s" in str(e.value)
    assert len(docker.probes) == 1 and docker.runs == []


def test_requirements_that_never_finish_installing_reject_the_submission_at_build(
    spec, docker, base, ref, tmp_path
):
    """What the requirements do at install time is the submission's, like what its policy does
    at start: a build that is not done within its timeout is a rejection, not a wait."""
    root = write_policy_repo(tmp_path / "repo", requirements="requirements.txt")
    (root / "requirements.txt").write_text("./stalls-forever\n")
    docker.images["x:y"] = FAKE_BASE_DIGEST
    docker.build_seconds = 100.0
    with pytest.raises(SubmissionRejected) as info:
        build_submission_image(
            docker, spec, root, check_repository(root, spec), ref, base, timeout_s=25.0
        )
    assert info.value.step == "build"
    assert info.value.reason == "installing requirements.txt did not finish within 25s"
    assert docker.build_timeouts == [25.0] and docker.runs == []
    # Given the time, it builds.
    built = build_submission_image(
        docker, spec, root, check_repository(root, spec), ref, base, timeout_s=200.0
    )
    assert built.tag == submission_tag(ref)
