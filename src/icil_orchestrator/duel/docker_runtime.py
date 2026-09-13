"""The duel's runtime seam, mapped onto the submission sandbox (`icil_orchestrator.submissions`).

This module is the only place a duel touches the submission code, so a change there changes this
adapter and nothing else in `duel/`:

- `resolve` and `fetch`: `HubFetcher` (or `LocalFetcher` for a repository mapped with `--local`),
  into the content-addressed `RepoCache`.
- `prepare`: `check_repository` (the manifest as a plain file), `build_submission_image` from the
  pinned base by digest, and a health check - a `PolicyContainer` started under
  `spec.submission.sandbox` that must answer `hello` within `budgets.policy_start_seconds`, then
  removed.
- `serve`: one `PolicyContainer` per unit, named `icil-duel-<key>-<random>`, with a socket directory
  of its own under the system temporary directory (a Unix socket path must stay short) and nothing
  else mounted: not the store, not the prompts, not the other side. It is listening when `serve`
  yields; the benchmark subprocess says `hello`. The container is removed when the unit is over,
  and its log is copied into the unit's directory first.

The submission code's two errors become the seam's: `SubmissionRejected` is `SubmissionRefused`
(the submission's own fault, with its step and reason), `SubmissionError` is `RuntimeUnavailable`.
"""

from __future__ import annotations

import os
import secrets
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..ids import SubmissionRef
from ..submissions import SubmissionError, SubmissionRejected
from ..submissions.checks import check_repository
from ..submissions.container import AUTHKEY_ENV, PolicyContainer
from ..submissions.docker import Docker
from ..submissions.fetch import HubFetcher, LocalFetcher, RepoCache
from ..submissions.image import base_image, build_submission_image
from .runtime import (
    FetchedSubmission,
    PolicyDied,
    PreparedSubmission,
    RuntimeUnavailable,
    ServedPolicy,
    SubmissionRefused,
)

CONTAINER_PREFIX = "icil-duel"
#: In the unit's directory once the unit is over: the server's log and what the policy printed.
LOG_FILE = "policy.log"
#: The most of a container's log copied out: the policy writes it and nothing else bounds it.
MAX_LOG_BYTES = 16 << 20
POLL_S = 0.1
#: How long a container whose client has gone gets to stop on its own before its state is read.
EXIT_GRACE_S = 10.0


class DockerPolicyRuntime:
    """Submissions fetched, built and served by the sandbox, one container per unit."""

    name = "docker"

    def __init__(
        self,
        spec: Any,
        docker: Docker | None = None,
        *,
        cache_dir: str | os.PathLike[str],
        local: Mapping[str, str | os.PathLike[str]] | None = None,
        base_digest: str | None = None,
        gpus: int | None = None,
        hub_api: Any = None,
        build_timeout_s: float | None = None,
    ) -> None:
        self.spec = spec
        self.docker = docker if docker is not None else Docker()
        self.cache = RepoCache(cache_dir, int(spec.submission["max_repo_bytes"]))
        self.local = {str(k): Path(v).resolve() for k, v in (local or {}).items()}
        self.base_digest = base_digest
        self.gpus = gpus
        self.hub_api = hub_api
        self.build_timeout_s = build_timeout_s
        self.start_timeout_s = float(spec.budgets["policy_start_seconds"])

    # -- the seam ---------------------------------------------------------------------------

    def resolve(self, repo: str, revision: str) -> SubmissionRef:
        with _mapped():
            return self._fetcher(repo).resolve(repo, revision).ref

    def fetch(self, ref: SubmissionRef, *, workdir: Path) -> FetchedSubmission:
        with _mapped():
            fetcher = self._fetcher(ref.repo)
            resolved = fetcher.resolve(ref.repo, ref.revision)
            fetched = fetcher.fetch(resolved)
        return FetchedSubmission(
            ref=ref,
            commit=resolved.sha,
            root=fetched.root,
            detail={"bytes": fetched.bytes, "files": fetched.files, "cached": fetched.cached},
        )

    def prepare(self, fetched: FetchedSubmission, *, workdir: Path) -> PreparedSubmission:
        with _mapped():
            manifest = check_repository(fetched.root, self.spec)
            base = base_image(self.spec, self.base_digest)
            built = build_submission_image(
                self.docker,
                self.spec,
                fetched.root,
                manifest,
                fetched.ref,
                base,
                timeout_s=self.build_timeout_s,
            )
            with self._container(built.tag, fetched.ref, Path(workdir)) as container:
                reply = container.hello(self.start_timeout_s)
        return PreparedSubmission(
            ref=fetched.ref,
            commit=fetched.commit,
            base_image_digest=base.digest,
            image=built.image_id,
            policy=manifest.policy,
            action_type=reply.get("action_type"),
            handle=built.tag,
        )

    @contextmanager
    def serve(self, prepared: PreparedSubmission, *, workdir: Path) -> Iterator[ServedPolicy]:
        with self._container(str(prepared.handle), prepared.ref, Path(workdir)) as container:
            try:
                container.start()
            except SubmissionError as exc:
                raise PolicyDied(f"the policy container did not start: {exc}") from None
            self._wait_listening(container)
            served = ServedPolicy(
                address=str(container.socket_path),
                authkey_env=AUTHKEY_ENV,
                env={AUTHKEY_ENV: container.authkey.hex()},
                log_file=Path(workdir) / LOG_FILE,
                died=lambda: self._died(container),
            )
            self._started(container, served)
            yield served

    # -- containers -------------------------------------------------------------------------

    def _fetcher(self, repo: str) -> Any:
        if repo in self.local:
            return LocalFetcher(self.cache, self.local[repo])
        return HubFetcher(self.cache, api=self.hub_api)

    @contextmanager
    def _container(
        self, image: str, ref: SubmissionRef, workdir: Path
    ) -> Iterator[PolicyContainer]:
        """A container for `image` with a private socket directory; removed, its log kept."""
        workdir.mkdir(parents=True, exist_ok=True)
        sockets = Path(tempfile.mkdtemp(prefix=f"{CONTAINER_PREFIX}-"))
        container = PolicyContainer(
            self.spec,
            self.docker,
            image,
            name=f"{CONTAINER_PREFIX}-{ref.key}-{secrets.token_hex(3)}",
            socket_dir=sockets,
            gpus=self.gpus,
        )
        try:
            yield container
        finally:
            try:
                container.close()
            finally:
                _copy_log(container.log_path, workdir / LOG_FILE)
                shutil.rmtree(sockets, ignore_errors=True)

    def _wait_listening(self, container: PolicyContainer) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        while not container.socket_path.exists():
            state = self.docker.state(container.name)
            if not state.running:
                error = f": {state.error}" if state.error else ""
                raise PolicyDied(
                    f"the policy container exited ({state.exit_code}) before listening{error}"
                )
            if time.monotonic() >= deadline:
                raise PolicyDied(
                    f"the policy container did not listen within {self.start_timeout_s:g}s"
                )
            time.sleep(POLL_S)

    def _started(self, container: PolicyContainer, served: ServedPolicy) -> None:
        """Called once a unit's container listens. The tests stand in here to kill one."""

    def _died(self, container: PolicyContainer) -> str | None:
        """Why the container died underneath its unit: any end but a clean exit after its client."""
        deadline = time.monotonic() + EXIT_GRACE_S
        state = self.docker.state(container.name)
        while state.running and time.monotonic() < deadline:
            time.sleep(POLL_S)
            state = self.docker.state(container.name)
        if state.running or state.exit_code == 0:
            return None
        error = f": {state.error}" if state.error else ""
        how = "was killed (137: out of memory, or a kill)" if state.exit_code == 137 else "exited"
        return f"the policy container {how} ({state.exit_code}){error}"


@contextmanager
def _mapped() -> Iterator[None]:
    try:
        yield
    except SubmissionRejected as exc:
        raise SubmissionRefused(exc.step, exc.reason) from None
    except SubmissionError as exc:
        raise RuntimeUnavailable(str(exc)) from None


def _copy_log(source: Path, target: Path) -> None:
    """Copy at most `MAX_LOG_BYTES` of a log the policy could have replaced with a link or a pipe:
    only a regular file is read, and nothing is followed."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(source, flags)
    except OSError:
        return
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return
        with open(fd, "rb", closefd=False) as fh, open(target, "ab") as out:
            out.write(fh.read(MAX_LOG_BYTES))
    except OSError:
        pass
    finally:
        os.close(fd)
