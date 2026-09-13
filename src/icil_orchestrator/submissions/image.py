"""The pinned base image, and a submission's own image built from it.

**The base** is `docker/policy-base/Dockerfile`, built with the repository root as its context
and referred to by digest - the image id Docker computes over its configuration and layers,
`sha256:<hex>` - which `spec.submission.base_image.digest` pins and every event record names.

**Why a tag stands in for the digest in `FROM`.** BuildKit resolves `FROM name@sha256:...`
against a registry's manifest digest, which a locally built image does not have, and refuses a
bare image id. So the base is tagged `<name>:<hex of the digest>`, a tag nothing else writes, and
the id behind that tag is checked against the digest immediately before every build. The record
names the digest, never the tag.

**A submission's image** is its checkout copied to `/submission` and its requirements installed,
with network, at build time - and nothing else. The image is tagged by the submission's key and
commit sha, so the same submission builds to the same tag and Docker's cache does the rest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icil_policy.manifest import Manifest

from ..ids import SubmissionRef
from ..spec import IMAGE_DIGEST_RE
from .docker import BuildFailed, BuildTimedOut, Docker
from .errors import SubmissionError, SubmissionRejected

#: The base image's Dockerfile, relative to the repository root that is its build context.
BASE_DOCKERFILE = Path("docker") / "policy-base" / "Dockerfile"
#: Where a submission's checkout lives inside its image.
SUBMISSION_DIR = "/submission"
#: The repository every submission image is tagged under.
SUBMISSION_IMAGE = "icil-submission"
#: Build arguments the base Dockerfile takes, from the spec's sandbox user.
UID_ARG, GID_ARG = "POLICY_UID", "POLICY_GID"


@dataclass(frozen=True)
class BaseImage:
    name: str
    digest: str

    @property
    def tag(self) -> str:
        """The local reference that stands for the digest; see the module docstring."""
        return f"{self.name}:{self.digest.partition(':')[2]}"


@dataclass(frozen=True)
class BuiltImage:
    tag: str
    image_id: str
    base: BaseImage
    seconds: float


def sandbox_user(spec: Any) -> tuple[int, int]:
    """`(uid, gid)` of `spec.submission.sandbox.user`, which must be numeric: the socket
    directory on the host is owned by it, and a name means nothing outside the container."""
    user = str(spec.submission["sandbox"]["user"])
    uid, _, gid = user.partition(":")
    if not uid.isdigit() or (gid and not gid.isdigit()):
        raise SubmissionError(f"submission.sandbox.user must be numeric uid[:gid], not {user!r}")
    return int(uid), int(gid) if gid else int(uid)


def base_image(spec: Any, digest: str | None = None) -> BaseImage:
    """The base the spec pins, or the `digest` given in its place while the spec's is null."""
    pinned = spec.submission["base_image"]
    chosen = digest or pinned.get("digest")
    if not chosen:
        raise SubmissionError(
            "no base image digest: spec.submission.base_image.digest is null and none was given; "
            "build one with `icil-orchestrator submission build-base` and pass --base-image"
        )
    if not IMAGE_DIGEST_RE.match(chosen):
        raise SubmissionError(f"{chosen!r} is not an image digest (sha256:<64 hex>)")
    return BaseImage(name=str(pinned["name"]), digest=chosen)


def build_base_image(
    docker: Docker, spec: Any, repo_root: Path, *, timeout_s: float | None = None
) -> BuiltImage:
    """Build `docker/policy-base/Dockerfile` with `repo_root` as its context; tag the result
    `<name>:<hex>` and `<name>:latest`. The digest is the id Docker gives it."""
    dockerfile = repo_root / BASE_DOCKERFILE
    try:
        text = dockerfile.read_text(encoding="utf-8")
    except OSError as exc:
        raise SubmissionError(f"cannot read {dockerfile}: {exc}") from None
    name = str(spec.submission["base_image"]["name"])
    uid, gid = sandbox_user(spec)
    started = time.monotonic()
    image_id = docker.build(
        repo_root,
        text,
        tag=f"{name}:latest",
        build_args={UID_ARG: str(uid), GID_ARG: str(gid)},
        timeout_s=timeout_s,
    )
    base = BaseImage(name=name, digest=image_id)
    docker.tag(image_id, base.tag)
    return BuiltImage(
        tag=base.tag, image_id=image_id, base=base, seconds=round(time.monotonic() - started, 3)
    )


def ensure_base(docker: Docker, base: BaseImage) -> str:
    """The tag that stands for `base.digest` on this host, checked to be exactly that image."""
    if docker.image_id(base.tag) == base.digest:
        return base.tag
    found = docker.find_image(base.digest)
    if found is None:
        raise SubmissionError(
            f"base image {base.digest} is not on this host; build it with "
            "`icil-orchestrator submission build-base` or load it, then check its digest"
        )
    docker.tag(base.digest, base.tag)
    if docker.image_id(base.tag) != base.digest:
        raise SubmissionError(f"could not tag {base.digest} as {base.tag}")
    return base.tag


def submission_tag(ref: SubmissionRef) -> str:
    return f"{SUBMISSION_IMAGE}:{ref.key}-{ref.revision}"


def submission_dockerfile(base: BaseImage, manifest: Manifest, spec: Any) -> str:
    """The whole of a submission's Dockerfile: the base, the checkout, its requirements."""
    uid, gid = sandbox_user(spec)
    lines = [
        f"# {base.digest}",
        f"FROM {base.tag}",
        f"COPY --chown={uid}:{gid} . {SUBMISSION_DIR}",
    ]
    if manifest.requirements is not None:
        lines.append(
            f"RUN python -m pip install --no-cache-dir -r {SUBMISSION_DIR}/{manifest.requirements}"
        )
    return "\n".join(lines) + "\n"


def build_submission_image(
    docker: Docker,
    spec: Any,
    checkout: Path,
    manifest: Manifest,
    ref: SubmissionRef,
    base: BaseImage,
    *,
    timeout_s: float | None = None,
) -> BuiltImage:
    """`checkout` as an image FROM `base`, tagged by `ref`. A build that fails - the requirements
    do not install - or is not done within `timeout_s` is a rejection with the reason: what the
    requirements do at install time is the submission's, like what its policy does at start. A
    base that is not there is not."""
    ensure_base(docker, base)
    dockerfile = submission_dockerfile(base, manifest, spec)
    tag = submission_tag(ref)
    started = time.monotonic()
    installing = (
        f"installing {manifest.requirements}"
        if manifest.requirements is not None
        else "building the image"
    )
    try:
        image_id = docker.build(checkout, dockerfile, tag=tag, timeout_s=timeout_s)
    except BuildFailed as exc:
        raise SubmissionRejected("build", f"{installing} failed:\n{exc.log}") from None
    except BuildTimedOut as exc:
        raise SubmissionRejected(
            "build", f"{installing} did not finish within {exc.timeout_s:g}s"
        ) from None
    return BuiltImage(
        tag=tag, image_id=image_id, base=base, seconds=round(time.monotonic() - started, 3)
    )
