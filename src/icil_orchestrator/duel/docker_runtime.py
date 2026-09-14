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

Every container is labelled with the store and the run root it serves (`bind`), besides the
sandbox's own label, so that `reap` - called by the orchestrator on start, holding the store's
lock - can remove the containers an orchestrator of the same store killed outright left running,
and no one else's.

Whose a container's end is comes from `docker inspect`: a container that exited non-zero, or that
the kernel killed for its sandbox's memory limit (`State.OOMKilled`), ended by its policy's doing;
a container Docker no longer knows (removed from outside, or Docker itself gone) did not. `docker
run` failing is Docker's failure too. Nothing here removes a container before its end is read.
"""

from __future__ import annotations

import logging
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
from ..submissions.container import AUTHKEY_ENV, PolicyContainer
from ..submissions.docker import Docker
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

log = logging.getLogger(__name__)

CONTAINER_PREFIX = "icil-duel"
#: In the unit's directory once the unit is over: the server's log and what the policy printed.
LOG_FILE = "policy.log"
#: The most of a container's log copied out: the policy writes it and nothing else bounds it.
MAX_LOG_BYTES = 16 << 20
POLL_S = 0.1
#: How long a container whose client has gone gets to stop on its own before its state is read.
EXIT_GRACE_S = 10.0
#: The labels naming what a container was started for: the store's root and the run root.
STORE_LABEL = "icil.duel.store"
RUNS_LABEL = "icil.duel.runs"


class _Labelled:
    """A docker client that adds `labels` to every `docker run` and is otherwise the client."""

    def __init__(self, docker: Any, labels: Mapping[str, str]) -> None:
        self._docker = docker
        self._labels = dict(labels)

    def run(self, args: Any, *, env: Mapping[str, str] | None = None) -> str:
        extra = [a for k, v in sorted(self._labels.items()) for a in ("--label", f"{k}={v}")]
        return self._docker.run([*extra, *args], env=env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._docker, name)


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
        #: Put on every container this runtime starts; empty until `bind`.
        self.labels: dict[str, str] = {}

    # -- the seam ---------------------------------------------------------------------------

    def bind(self, *, store: Path, runs: Path) -> None:
        self.labels = {
            STORE_LABEL: str(Path(store).resolve()),
            RUNS_LABEL: str(Path(runs).resolve()),
        }

    def reap(self) -> list[str]:
        """Remove every `icil-duel-*` container labelled with this runtime's store and run root:
        with the store's lock held, none of them is a duel still running. Best effort: a Docker
        that cannot list them reaps nothing, and the duel that follows says why."""
        run = getattr(self.docker, "_run", None)
        if not self.labels or run is None:
            return []
        filters = [
            a for k, v in sorted(self.labels.items()) for a in ("--filter", f"label={k}={v}")
        ]
        try:
            done = run(
                ["ps", "--all", *filters, "--format", "{{.Names}}"], check=False, timeout_s=60
            )
        except SubmissionError as exc:
            log.warning("could not list the containers a killed orchestrator left: %s", exc)
            return []
        if done.returncode != 0:
            log.warning("could not list the containers a killed orchestrator left: %s", done.stderr)
            return []
        names = [n for n in done.stdout.split() if n.startswith(f"{CONTAINER_PREFIX}-")]
        for name in names:
            self.docker.remove(name)
        return names

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
            _Labelled(self.docker, self.labels),  # type: ignore[arg-type]
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
        while not container.socket_path.exists():
            state = self.docker.state(container.name)
            if not state.running:
                error = f": {state.error}" if state.error else ""
                if state.exit_code is None:
                    raise RuntimeUnavailable(f"the policy container is gone{error}")
                raise PolicyDied(
                    f"the policy container {self._how(container, state.exit_code)} "
                    f"before listening{error}"
                )
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
        return PolicyEnd(
            f"the policy container {self._how(container, state.exit_code)}{error}", POLICY
        )

    def _how(self, container: PolicyContainer, exit_code: int) -> str:
        if self._oom_killed(container.name):
            memory = int(self.spec.submission["sandbox"]["memory_bytes"])
            return f"was killed for going over its {memory} bytes of memory ({exit_code})"
        return f"exited ({exit_code})"

    def _oom_killed(self, name: str) -> bool:
        """`State.OOMKilled`, which `Docker.state` does not read; False when it cannot be told."""
        run = getattr(self.docker, "_run", None)
        if run is None:
            return False
        try:
            done = run(
                ["inspect", "--type", "container", "--format", "{{.State.OOMKilled}}", name],
                check=False,
                timeout_s=60,
            )
        except SubmissionError as exc:
            log.warning("could not ask docker whether %s ran out of memory: %s", name, exc)
            return False
        return done.returncode == 0 and done.stdout.strip() == "true"


@contextmanager
def _mapped() -> Iterator[None]:
    try:
        yield
    except SubmissionRejected as exc:
        raise SubmissionRefused(exc.step, exc.reason) from None
    except SubmissionError as exc:
        raise RuntimeUnavailable(str(exc)) from None
