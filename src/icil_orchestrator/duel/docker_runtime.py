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

A unit's container runs exactly as the sandbox runs any policy: `PolicyContainer.start` builds its
`docker run` (`run_argv`: the scratch tmpfs that may run code, `HOME` and the JIT caches in it, the
socket directory's own noexec tmpfs, the limits) and nothing here adds to it or wraps the client.

The submission code's two errors become the seam's: `SubmissionRejected` is `SubmissionRefused`
(the submission's own fault, with its step and reason), `SubmissionError` is `RuntimeUnavailable`.

Every container carries the sandbox's labels, the process that started it among them
(`submissions.container.Owner`), and every container's start removes those whose process is gone
(`reap_orphans`). `reap` does the same when the orchestrator starts, holding the store's lock, so
what an orchestrator killed outright left running is gone before any unit runs; a container a
live process holds is never touched.

Whose a container's end is comes from `Docker.state` (`docker inspect`): a container that exited
non-zero, or that the kernel killed for its sandbox's memory limit (`State.OOMKilled`), ended by its
policy's doing; a container Docker no longer knows (removed from outside, or Docker itself gone)
did not. `docker run` failing is Docker's failure too. Nothing here removes a container before its
end is read.
"""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..ids import SubmissionRef
from ..submissions import SubmissionError, SubmissionRejected
from ..submissions.checks import check_repository
from ..submissions.container import AUTHKEY_ENV, PolicyContainer, is_socket, reap_orphans
from ..submissions.docker import ContainerState, Docker
from ..submissions.fetch import HubFetcher, LocalFetcher, RepoCache
from ..submissions.image import base_image, build_submission_image
from .runtime import (
    HARNESS,
    POLICY,
    FetchedSubmission,
    PolicyDied,
    PolicyEnd,
    PreparedSubmission,
    RuntimeUnavailable,
    ServedPolicy,
    SubmissionRefused,
    copy_log,
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

    def bind(self, *, store: Path, runs: Path) -> None:
        """Nothing to mark: the sandbox labels every container with the process that starts it."""

    def reap(self) -> list[str]:
        """Remove every policy container whose owner process is gone, with its shared tmpfs
        (`reap_orphans`); the names removed. Best effort: a Docker that cannot list them reaps
        nothing, and the duel that follows says why."""
        return reap_orphans(self.docker)

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
                raise RuntimeUnavailable(f"docker could not start the container: {exc}") from None
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
                copy_log(container.log_path, workdir / LOG_FILE, MAX_LOG_BYTES)
                shutil.rmtree(sockets, ignore_errors=True)

    def _wait_listening(self, container: PolicyContainer) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        while not is_socket(container.socket_path):
            state = self.docker.state(container.name)
            if not state.running:
                error = f": {state.error}" if state.error else ""
                if state.exit_code is None:
                    raise RuntimeUnavailable(f"the policy container is gone{error}")
                raise PolicyDied(f"the policy container {self._how(state)} before listening{error}")
            if time.monotonic() >= deadline:
                raise PolicyDied(
                    f"the policy container did not listen within {self.start_timeout_s:g}s"
                )
            time.sleep(POLL_S)

    def _started(self, container: PolicyContainer, served: ServedPolicy) -> None:
        """Called once a unit's container listens. The tests stand in here to kill one."""

    def _died(self, container: PolicyContainer) -> PolicyEnd | None:
        """How the container ended underneath its unit: any end but a clean exit after its client.
        A container Docker cannot describe any more is the harness's; any other end, the policy's."""
        deadline = time.monotonic() + EXIT_GRACE_S
        state = self.docker.state(container.name)
        while state.running and time.monotonic() < deadline:
            time.sleep(POLL_S)
            state = self.docker.state(container.name)
        if state.running or state.exit_code == 0:
            return None
        error = f": {state.error}" if state.error else ""
        if state.exit_code is None:
            return PolicyEnd(f"the policy container is gone{error}", HARNESS)
        return PolicyEnd(f"the policy container {self._how(state)}{error}", POLICY)

    def _how(self, state: ContainerState) -> str:
        if state.oom_killed:
            memory = int(self.spec.submission["sandbox"]["memory_bytes"])
            return f"was killed for going over its {memory} bytes of memory ({state.exit_code})"
        return f"exited ({state.exit_code})"


@contextmanager
def _mapped() -> Iterator[None]:
    try:
        yield
    except SubmissionRejected as exc:
        raise SubmissionRefused(exc.step, exc.reason) from None
    except SubmissionError as exc:
        raise RuntimeUnavailable(str(exc)) from None
