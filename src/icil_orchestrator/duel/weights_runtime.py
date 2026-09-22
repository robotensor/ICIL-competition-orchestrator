"""A weights-only submission, served by the validator's own policy code: no sandbox, none needed.

A track whose spec says `submission.kind: weights` takes a Hugging Face repository holding the
weights of one architecture and nothing else (`submission.model`). Nothing in the repository is
code, and nothing of it is executed or unpickled: its one weights file is a safetensors file,
whose header is checked against the architecture's pinned template before a tensor is read, and
the policy that loads it is the validator's (`submission.model.policy`, installed in the policy
environment this runtime is given). That is why this runtime needs no Docker: the sandbox exists to
hold untrusted code, and here there is none.

The seam is `SubprocessPolicyRuntime`'s, with three steps replaced:

- `resolve` pins `repo@revision` to a commit on the Hub, which also lists the repository's files.
- `fetch` refuses a repository holding any file outside `submission.model.allowed_files`, or no
  weights file, and downloads the weights file alone into a cache addressed by the commit.
- `prepare` checks the weights file against the template (`checker`), writes a manifest naming the
  validator's policy class with the weights' absolute path, and has the policy answer `hello`,
  which builds the model and loads the weights: a file that passes the header check but does not
  load is refused here, before a unit is played.

`serve` is the subprocess runtime's: a fresh `python -m icil_policy.serve` per unit, in the policy
environment (`python`), with a random key that travels by variable name.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from icil_policy.client import RemotePolicy
from icil_policy.errors import PolicyUnavailable

from ..ids import SubmissionRef
from ..submissions import SubmissionError, SubmissionRejected
from ..submissions.fetch import HubFetcher, RepoCache
from ..submissions.resolve import REPO_TYPE, Resolved
from .local_runtime import SubprocessPolicyRuntime
from .runtime import (
    FetchedSubmission,
    PolicyDied,
    PreparedSubmission,
    RuntimeUnavailable,
    SubmissionRefused,
)

#: The manifest the runtime writes for the validator's policy class; never a submission's file.
MANIFEST_FILE = "icil.yaml"
#: Environment variables a served BPP policy needs beyond the subprocess allow-list: where the
#: Hugging Face and torch caches are (the CLIP backbone's weights are read from there once the
#: host has them), and whether the Hub may be asked at all.
POLICY_ENV_KEEP = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "BPP_RUNTIME_TEMPLATE",
)


@dataclass(frozen=True)
class WeightsCheck:
    """What the template check found: `ok` with no `errors`, and the weights' content hash."""

    ok: bool
    errors: tuple[str, ...] = ()
    weights_sha256: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def default_checker(weights: Path) -> WeightsCheck:
    """`bpp_runtime.check` on the weights file: the header against the pinned template, the size
    cap and the file's sha256. Imported here, lazily, because it is the only part of the model
    runtime a validator host needs, and it needs no torch."""
    from bpp_runtime import check as bpp_check

    report = bpp_check.check_weights(weights)
    return WeightsCheck(
        ok=bool(report.ok),
        errors=tuple(str(e) for e in report.errors),
        weights_sha256=report.weights_sha256,
        detail={
            key: value
            for key, value in {
                "architecture": getattr(report, "architecture", None),
                "param_count": getattr(report, "param_count", None),
                "bytes": getattr(report, "bytes", None),
            }.items()
            if value is not None
        },
    )


class WeightsPolicyRuntime(SubprocessPolicyRuntime):
    """Serve weights-only submissions with the validator's policy class. No sandbox: no code of a
    submission's ever runs (see the module docstring)."""

    name = "weights"
    env_keep = POLICY_ENV_KEEP

    def __init__(
        self,
        spec: Any,
        *,
        python: str = sys.executable,
        cache_root: str | os.PathLike[str],
        api: Any = None,
        download: Callable[..., Any] | None = None,
        checker: Callable[[Path], WeightsCheck] = default_checker,
        kwargs: dict[str, Any] | None = None,
        start_timeout_s: float | None = None,
        hello: bool = True,
    ) -> None:
        super().__init__(spec, {}, python=python, start_timeout_s=start_timeout_s)
        if spec.submission_kind != "weights":
            raise ValueError("the weights runtime serves a spec whose submission.kind is weights")
        self.model = spec.model
        self.weights_file = str(self.model["weights_file"])
        self.allowed = frozenset(str(f) for f in self.model["allowed_files"])
        self.cache = RepoCache(cache_root, int(spec.submission["max_repo_bytes"]))
        self.fetcher = HubFetcher(self.cache, api=api, download=download or self._download)
        self.checker = checker
        #: Extra constructor keywords for the policy class (a device, say); never the weights path,
        #: which the runtime sets.
        self.kwargs = dict(kwargs or {})
        self.hello = hello

    # -- the seam ---------------------------------------------------------------------------

    def resolve(self, repo: str, revision: str) -> SubmissionRef:
        with _mapped():
            return self.fetcher.resolve(repo, revision).ref

    def fetch(self, ref: SubmissionRef, *, workdir: Path) -> FetchedSubmission:
        with _mapped():
            resolved = self.fetcher.resolve(ref.repo, ref.revision)
            names = sorted(f.path for f in resolved.files)
            extra = [name for name in names if name not in self.allowed]
            if extra:
                shown = ", ".join(extra[:5]) + (", ..." if len(extra) > 5 else "")
                raise SubmissionRejected(
                    "fetch",
                    f"{ref.entry} holds {len(extra)} file(s) a weights submission may not "
                    f"({shown}); it may hold only {', '.join(sorted(self.allowed))}",
                )
            if self.weights_file not in names:
                raise SubmissionRejected("fetch", f"{ref.entry} holds no {self.weights_file}")
            # Only the weights file is downloaded: a README is allowed to be there, and read by
            # nobody here.
            weights_only = Resolved(
                repo=resolved.repo,
                revision=resolved.revision,
                sha=resolved.sha,
                files=tuple(f for f in resolved.files if f.path == self.weights_file),
            )
            fetched = self.fetcher.fetch(weights_only)
        return FetchedSubmission(
            ref=ref,
            commit=resolved.sha,
            root=fetched.root,
            detail={"bytes": fetched.bytes, "files": names, "cached": fetched.cached},
        )

    def prepare(self, fetched: FetchedSubmission, *, workdir: Path) -> PreparedSubmission:
        weights = (fetched.root / self.weights_file).resolve()
        if not weights.is_file():
            raise SubmissionRefused("fetch", f"{self.weights_file} is not in the fetched checkout")
        try:
            report = self.checker(weights)
        except Exception as exc:  # noqa: BLE001 - a checker that cannot run is the harness's
            raise RuntimeUnavailable(f"the weights check could not run: {exc}") from exc
        if not report.ok:
            shown = "; ".join(report.errors[:8]) or "no reason given"
            raise SubmissionRefused("check", f"{self.weights_file}: {shown}")
        manifest = self._write_manifest(Path(workdir), weights)
        action_type = "ee"
        if self.hello:
            try:
                with self._serve(manifest, Path(workdir)) as (served, key, live_log):
                    with RemotePolicy(
                        served.address, key, timeout_s=self.start_timeout_s, log_file=live_log
                    ) as policy:
                        reply = policy.hello()
                action_type = reply.get("action_type") or action_type
            except PolicyDied as exc:
                raise SubmissionRefused("start", str(exc)) from None
            except PolicyUnavailable as exc:
                raise SubmissionRefused("hello", str(exc)) from None
        return PreparedSubmission(
            ref=fetched.ref,
            commit=fetched.commit,
            base_image_digest=None,
            # A weights submission's own content address stands where a code submission's image
            # would: what exactly was served.
            image=f"sha256:{report.weights_sha256}" if report.weights_sha256 else None,
            policy=str(self.model["policy"]),
            action_type=action_type,
            handle=manifest,
        )

    # -- helpers ----------------------------------------------------------------------------

    def _write_manifest(self, workdir: Path, weights: Path) -> Path:
        """The validator's own `icil.yaml`, beside nothing of the submission's: its policy class
        and the weights file's absolute path (the server changes directory before building)."""
        directory = workdir / "manifest"
        directory.mkdir(parents=True, exist_ok=True)
        kwargs = {**self.kwargs, "weights": str(weights)}
        # YAML is a superset of JSON: a JSON object is a valid manifest, and needs no YAML writer.
        doc = {"api": 1, "policy": str(self.model["policy"]), "kwargs": kwargs}
        path = directory / MANIFEST_FILE
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return path

    def _download(self, repo: str, *, revision: str, repo_type: str, local_dir: str) -> None:
        from huggingface_hub import hf_hub_download

        if repo_type != REPO_TYPE:
            raise SubmissionError(f"a submission is a {REPO_TYPE} repository, not a {repo_type}")
        hf_hub_download(
            repo,
            self.weights_file,
            revision=revision,
            repo_type=repo_type,
            local_dir=local_dir,
        )


@contextmanager
def _mapped() -> Iterator[None]:
    """The submission code's two errors as the seam's (as the Docker runtime maps them)."""
    try:
        yield
    except SubmissionRejected as exc:
        raise SubmissionRefused(exc.step, exc.reason) from None
    except SubmissionError as exc:
        raise RuntimeUnavailable(str(exc)) from None
