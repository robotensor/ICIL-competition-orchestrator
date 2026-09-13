"""A submission taken through every step, each reported: resolve, fetch, manifest, build, start,
hello. What `icil-orchestrator submission check` runs, and what a duel runs for each side before
a unit is played.

The report says where a submission stopped and why. A step that rejects the submission is the
entry's own doing and its reason can be published; a step that errors is the harness's, and the
submission is not judged. Whatever happened, the container is removed before this returns.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checks import check_repository
from .container import PolicyContainer
from .docker import Docker
from .errors import SubmissionError, SubmissionRejected
from .image import base_image, build_submission_image

STEPS = ("resolve", "fetch", "manifest", "build", "start", "hello")
CONTAINER_PREFIX = "icil-policy"


@dataclass
class Step:
    name: str
    #: pending | ok | rejected | error | skipped
    status: str = "pending"
    seconds: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "seconds": self.seconds,
            "detail": self.detail,
        }


@dataclass
class SubmissionReport:
    repo: str
    revision: str
    steps: list[Step] = field(default_factory=lambda: [Step(name) for name in STEPS])
    sha: str | None = None
    key: str | None = None
    checkout_bytes: int | None = None
    checkout_files: int | None = None
    policy: str | None = None
    base_image: str | None = None
    image_tag: str | None = None
    image: str | None = None
    build_seconds: float | None = None
    listening_after_s: float | None = None
    hello: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return all(step.status == "ok" for step in self.steps)

    @property
    def verdict(self) -> str:
        """accepted, rejected or error."""
        for step in self.steps:
            if step.status in ("rejected", "error"):
                return step.status
        return "accepted" if self.ok else "error"

    @property
    def failed_step(self) -> Step | None:
        return next((s for s in self.steps if s.status in ("rejected", "error")), None)

    def step(self, name: str) -> Step:
        return next(s for s in self.steps if s.name == name)

    def side(self) -> dict[str, Any]:
        """What an event record carries for this submission as one side of a duel: the resolved
        commit, the base image digest, the image built from it and how the policy answered."""
        return {
            "key": self.key,
            "repo": self.repo,
            "revision": self.sha,
            "base_image": self.base_image,
            "image": self.image,
            "policy": self.policy,
            "action_type": (self.hello or {}).get("action_type"),
            "protocol": (self.hello or {}).get("protocol"),
            "start_seconds": self.listening_after_s,
            "verdict": self.verdict,
            "reason": self.failed_step.detail if self.failed_step else None,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "sha": self.sha,
            "key": self.key,
            "verdict": self.verdict,
            "steps": [s.as_dict() for s in self.steps],
            "checkout_bytes": self.checkout_bytes,
            "checkout_files": self.checkout_files,
            "policy": self.policy,
            "base_image": self.base_image,
            "image_tag": self.image_tag,
            "image": self.image,
            "build_seconds": self.build_seconds,
            "listening_after_s": self.listening_after_s,
            "hello": self.hello,
            "side": self.side(),
        }


def check_submission(
    spec: Any,
    repo: str,
    revision: str,
    *,
    fetcher: Any,
    docker: Docker,
    work_dir: Path,
    base_digest: str | None = None,
    gpus: int | None = None,
    build_timeout_s: float | None = None,
) -> SubmissionReport:
    """Every step in order, stopping at the first that fails; the container is gone on return.

    `fetcher` is a `HubFetcher` or a `LocalFetcher`; `work_dir` holds the socket directory and
    the policy's log, which are left for the caller to look at or remove.
    """
    report = SubmissionReport(repo=repo, revision=revision)
    container: PolicyContainer | None = None
    current = report.steps[0]

    @contextmanager
    def step(name: str) -> Iterator[Step]:
        nonlocal current
        current = report.step(name)
        started = time.monotonic()
        try:
            yield current
        finally:
            current.seconds = round(time.monotonic() - started, 3)

    try:
        with step("resolve") as s:
            resolved = fetcher.resolve(repo, revision)
            ref = resolved.ref
            report.sha, report.key = ref.revision, ref.key
            s.detail = f"{repo}@{revision} -> {ref.revision} (key {ref.key})"
            s.status = "ok"
        with step("fetch") as s:
            fetched = fetcher.fetch(resolved)
            report.checkout_bytes, report.checkout_files = fetched.bytes, fetched.files
            s.detail = f"{fetched.bytes} bytes, {fetched.files} files"
            s.detail += " (cached)" if fetched.cached else f" -> {fetched.root}"
            s.status = "ok"
        with step("manifest") as s:
            manifest = check_repository(fetched.root, spec)
            report.policy = manifest.policy
            s.detail = f"policy {manifest.policy}"
            if manifest.requirements is not None:
                s.detail += f", requirements {manifest.requirements}"
            s.status = "ok"
        with step("build") as s:
            base = base_image(spec, base_digest)
            report.base_image = base.digest
            built = build_submission_image(
                docker, spec, fetched.root, manifest, ref, base, timeout_s=build_timeout_s
            )
            report.image_tag, report.image = built.tag, built.image_id
            report.build_seconds = built.seconds
            s.detail = f"{built.tag} = {built.image_id} FROM {base.name} {base.digest}"
            s.status = "ok"
        with step("start") as s:
            name = f"{CONTAINER_PREFIX}-{ref.key}-{secrets.token_hex(3)}"
            container = PolicyContainer(
                spec, docker, built.tag, name=name, socket_dir=work_dir / "policy", gpus=gpus
            )
            container.start()
            shared = "a tmpfs" if container.bounded else "a plain directory, unbounded (not root)"
            s.detail = f"container {name}, socket directory {container.socket_dir} ({shared})"
            s.status = "ok"
        with step("hello") as s:
            budget = float(spec.budgets["policy_start_seconds"])
            reply = container.hello(budget)
            report.hello = dict(reply)
            report.listening_after_s = container.listening_after_s
            s.detail = (
                f"protocol {reply.get('protocol')}, action_type {reply.get('action_type')}, "
                f"policy {reply.get('policy')}; listening after {container.listening_after_s}s"
            )
            s.status = "ok"
    except SubmissionRejected as exc:
        failed = report.step(exc.step) if exc.step in STEPS else current
        failed.status, failed.detail = "rejected", exc.reason
        if failed is not current:
            current.status = "skipped"
    except SubmissionError as exc:
        current.status, current.detail = "error", str(exc)
    finally:
        if container is not None:
            container.close()
    for pending in report.steps:
        if pending.status == "pending":
            pending.status = "skipped"
    return report
