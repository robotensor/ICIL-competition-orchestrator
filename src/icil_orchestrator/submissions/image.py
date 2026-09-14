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

**Copied, not mounted.** The checkout is in the image rather than bind-mounted read-only at run
time: the requirements install needs it at build time (a requirement may be the repository's own
package), the image id recorded with a side is then exactly the code that ran, and the container
mounts nothing of the host but its socket directory. Under `--read-only` `/submission` is as
unwritable as a read-only mount. The cost is disk: a checkout of up to `max_repo_bytes` is also a
layer, so images stay until `submission prune` (`prune_submission_images`) removes them.

**Whose failure a failed build is.** The requirements do not install: the submission's doing, a
rejection. Or pip could not reach its index: the network's or the index's, an error to try again.
The build's log cannot tell the two apart, because the submission writes it: a `setup.py` can
print pip's words for a broken connection. So the log decides nothing. After a build fails or
runs out of time the orchestrator runs its own probe (`probe_index`): a build from the same base
with nothing of the submission in it, never cached, in which pip downloads `pip` from the index it
is configured with. If the probe gets through, the index was reachable and the failure is a
rejection, whatever the log says. If it does not, nobody can tell, and the build is the harness's
error. The probe starts after the submission's build has ended, so nothing of the submission runs
beside it. Requirements that name an index of their own (`--index-url`) are the submission's to
keep reachable. What a submission can still do is make the index refuse this host during its own
build, and so turn its rejection into an error: that gains it a retry, never a verdict or a run.
"""

from __future__ import annotations

import secrets
import tempfile
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
#: The repository the index probe's throwaway image is tagged under, removed once it is built.
INDEX_PROBE_IMAGE = "icil-index-probe"
#: What the probe downloads: pip itself, which the index pip is configured with has.
INDEX_PROBE_REQUIREMENT = "pip"
#: How long the probe may take: pip's own retries against a network that drops packets, and more.
INDEX_PROBE_TIMEOUT_S = 180.0
#: How much of the probe's log an error quotes.
INDEX_PROBE_TAIL_CHARS = 500


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


@dataclass(frozen=True)
class IndexProbe:
    """What `probe_index` found: whether pip in the base reached its index, and what it said."""

    reachable: bool
    detail: str
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


def index_probe_dockerfile(base: BaseImage) -> str:
    """The probe's whole Dockerfile: the base and one pip download, nothing of any submission."""
    download = (
        "python -m pip download --no-deps --only-binary :all: --no-cache-dir "
        f"--dest /tmp/icil-index-probe {INDEX_PROBE_REQUIREMENT}"
    )
    return (
        f"# the orchestrator's index probe, from {base.digest}\n"
        f"FROM {base.tag}\n"
        f"RUN {download} && rm -rf /tmp/icil-index-probe\n"
    )


def probe_index(
    docker: Docker, base: BaseImage, *, timeout_s: float = INDEX_PROBE_TIMEOUT_S
) -> IndexProbe:
    """Whether pip, in a build from `base` with an empty context, reaches the index it is
    configured with: the orchestrator's own look at the network a submission's build had, which
    no submission can write to. Never cached, so every probe goes to the index; its image is
    removed. Docker itself failing is a `SubmissionError`, as anywhere else."""
    ensure_base(docker, base)
    tag = f"{INDEX_PROBE_IMAGE}:{secrets.token_hex(6)}"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="icil-index-probe-") as context:
        try:
            docker.build(
                Path(context),
                index_probe_dockerfile(base),
                tag=tag,
                timeout_s=timeout_s,
                no_cache=True,
            )
        except BuildFailed as exc:
            detail = exc.log[-INDEX_PROBE_TAIL_CHARS:]
            return IndexProbe(False, detail, round(time.monotonic() - started, 3))
        except BuildTimedOut as exc:
            detail = f"pip download {INDEX_PROBE_REQUIREMENT} did not finish in {exc.timeout_s:g}s"
            return IndexProbe(False, detail, round(time.monotonic() - started, 3))
    docker.remove_image(tag)
    seconds = round(time.monotonic() - started, 3)
    detail = f"pip download {INDEX_PROBE_REQUIREMENT} worked in {seconds:g}s"
    return IndexProbe(True, detail, seconds)


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
    do not install - or is not done within `timeout_s` is a rejection with the reason, once
    `probe_index` has reached the index: what the requirements do at install time is the
    submission's, like what its policy does at start. A base that is not there, or a probe that
    cannot reach the index either, is the harness's error; what the build printed decides
    nothing (see the module docstring)."""
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
        failed, log = f"{installing} failed", f":\n{exc.log}"
    except BuildTimedOut as exc:
        failed, log = f"{installing} did not finish within {exc.timeout_s:g}s", ""
    else:
        return BuiltImage(
            tag=tag, image_id=image_id, base=base, seconds=round(time.monotonic() - started, 3)
        )
    probe = probe_index(docker, base)
    if not probe.reachable:
        raise SubmissionError(
            f"{failed}, and the orchestrator's own probe could not reach the index either, so "
            f"the failure is not known to be the submission's; try again later. The probe: "
            f"{probe.detail}\nThe build{log or ': (no log, it was stopped)'}"
        )
    raise SubmissionRejected("build", failed + log)


def prune_submission_images(docker: Docker) -> tuple[list[str], dict[str, str]]:
    """Remove every `SUBMISSION_IMAGE` tag no container was made from: `(removed, {kept: why})`.
    Never forced, so an image a container still uses - a check or a duel in progress, or a
    leftover `reap_orphans` has not reached - is kept and says why."""
    removed: list[str] = []
    kept: dict[str, str] = {}
    for ref in docker.image_refs(SUBMISSION_IMAGE):
        why = docker.remove_image(ref, force=False)
        if why:
            kept[ref] = why
        else:
            removed.append(ref)
    return removed, kept
