"""The pinned base image and a submission's image built from it, with Docker stood in for."""

from __future__ import annotations

from pathlib import Path

import pytest

from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.submissions import SubmissionError, SubmissionRejected
from icil_orchestrator.submissions.checks import check_repository
from icil_orchestrator.submissions.image import (
    BASE_DOCKERFILE,
    BaseImage,
    base_image,
    build_base_image,
    build_submission_image,
    ensure_base,
    sandbox_user,
    submission_dockerfile,
    submission_tag,
)
from submission_helpers import FAKE_BASE_DIGEST, SHA_A, FakeDocker, write_policy_repo

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
