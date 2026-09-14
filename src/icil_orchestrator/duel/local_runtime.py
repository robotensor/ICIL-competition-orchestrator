"""A submission's policy served straight from a local directory, on this host: NO SANDBOX.

`icil-orchestrator duel --runtime local` and the pure tests use it. It is `python -m
icil_policy.serve` in a subprocess of this interpreter, on a Unix socket, with the directory
mapped by `--local REPO=DIR` as the submission's checkout: nothing is fetched, no image is built,
the policy's requirements are whatever this environment holds, and the policy runs with every
permission the orchestrator has. That is fine for the example policies and for code you wrote;
it is never how a competitor's submission runs (see `docker_runtime`).

Everything else is as the sandbox does it, so a duel behaves the same on either: the manifest is
checked and the policy must answer `hello` before a unit is played, each unit gets a fresh server,
the authkey is random per server and travels by variable name only, and a server that dies
underneath its unit is reported.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from icil_policy.client import RemotePolicy
from icil_policy.errors import ManifestError, PolicyUnavailable
from icil_policy.logs import tail
from icil_policy.manifest import load as load_manifest

from ..benchmarks.subprocess_runner import benchmark_environment
from ..ids import SubmissionRef, is_commit_sha
from .orphans import Ledger
from .runtime import (
    POLICY,
    FetchedSubmission,
    PolicyDied,
    PolicyEnd,
    PreparedSubmission,
    RuntimeUnavailable,
    ServedPolicy,
    SubmissionRefused,
)

#: The variable a served policy's key travels in, as in the sandbox.
AUTHKEY_ENV = "ICIL_POLICY_AUTHKEY"
AUTHKEY_BYTES = 32
SOCKET_FILE = "policy.sock"
#: In the unit's directory: the server's log and everything the policy printed.
LOG_FILE = "policy.log"
POLL_S = 0.05
#: How long a server whose client has gone gets to exit on its own before its status is read.
EXIT_GRACE_S = 5.0
#: What a local directory holds that is not the submission.
IGNORED = frozenset({".git", "__pycache__"})


def local_directories(pairs: Mapping[str, str | os.PathLike[str]]) -> dict[str, Path]:
    return {str(k): Path(v).resolve() for k, v in pairs.items()}


def tree_hash(directory: Path) -> str:
    """A 40-hex address of a directory's files and bytes, standing in for a commit sha."""
    digest = hashlib.sha1()
    paths = []
    for root, dirs, files in os.walk(directory):  # never follows a link to a directory
        dirs[:] = [d for d in dirs if d not in IGNORED]
        paths += [Path(root) / name for name in (*files, *dirs)]
    for path in sorted(paths):
        rel = path.relative_to(directory)
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode):
            content = hashlib.sha1(path.read_bytes()).hexdigest()
        elif stat.S_ISLNK(info.st_mode):
            content = "link:" + os.readlink(path)
        else:
            continue
        digest.update(f"{rel.as_posix()}\0{content}\n".encode())
    return digest.hexdigest()


class SubprocessPolicyRuntime:
    """Serve local directories' policies as subprocesses. Development only: no sandbox."""

    name = "local"

    def __init__(
        self,
        spec: Any,
        local: Mapping[str, str | os.PathLike[str]],
        *,
        python: str = sys.executable,
        start_timeout_s: float | None = None,
    ) -> None:
        self.spec = spec
        self.local = local_directories(local)
        self.python = python
        self.start_timeout_s = float(
            start_timeout_s if start_timeout_s is not None else spec.budgets["policy_start_seconds"]
        )

    # -- the seam ---------------------------------------------------------------------------

    def bind(self, *, store: Path, runs: Path) -> None:
        """Nothing to label: a policy server here is a process group, in its unit's ledger."""

    def reap(self) -> list[str]:
        """Nothing beyond the ledgers, which the orchestrator reaps."""
        return []

    def directory(self, ref_or_repo: SubmissionRef | str) -> Path:
        repo = ref_or_repo if isinstance(ref_or_repo, str) else ref_or_repo.repo
        key = None if isinstance(ref_or_repo, str) else ref_or_repo.key
        found = self.local.get(repo) or (self.local.get(key) if key else None)
        if found is None:
            raise RuntimeUnavailable(
                f"no local directory for {repo}: the local runtime serves only what --local maps"
            )
        if not found.is_dir():
            raise RuntimeUnavailable(f"{found} is not a directory")
        return found

    def resolve(self, repo: str, revision: str) -> SubmissionRef:
        """A commit sha as given (the directory stands for that commit), or the directory's tree
        hash for a branch or tag name, which a local directory cannot resolve."""
        directory = self.directory(repo)
        sha = revision if is_commit_sha(revision) else tree_hash(directory)
        return SubmissionRef.resolved(repo, sha)

    def fetch(self, ref: SubmissionRef, *, workdir: Path) -> FetchedSubmission:
        directory = self.directory(ref)
        return FetchedSubmission(
            ref=ref,
            commit=ref.revision,
            root=directory,
            detail={"directory": str(directory), "tree_sha1": tree_hash(directory)},
        )

    def prepare(self, fetched: FetchedSubmission, *, workdir: Path) -> PreparedSubmission:
        name = str(self.spec.submission["manifest"])
        path = fetched.root / name
        try:
            manifest = load_manifest(path)
        except ManifestError as exc:
            raise SubmissionRefused("manifest", f"{name}: {'; '.join(exc.problems)}") from None
        wanted = int(self.spec.submission["manifest_api"])
        if manifest.api != wanted:
            raise SubmissionRefused(
                "manifest", f"{name}: api {manifest.api} is not this competition's {wanted}"
            )
        try:
            with self._serve(path, Path(workdir)) as (served, key):
                with RemotePolicy(
                    served.address, key, timeout_s=self.start_timeout_s, log_file=served.log_file
                ) as policy:
                    reply = policy.hello()
        except PolicyDied as exc:
            raise SubmissionRefused("start", str(exc)) from None
        except PolicyUnavailable as exc:
            raise SubmissionRefused("hello", str(exc)) from None
        return PreparedSubmission(
            ref=fetched.ref,
            commit=fetched.commit,
            base_image_digest=None,
            image=None,
            policy=manifest.policy,
            action_type=reply.get("action_type"),
            handle=path,
        )

    @contextmanager
    def serve(self, prepared: PreparedSubmission, *, workdir: Path) -> Iterator[ServedPolicy]:
        with self._serve(Path(prepared.handle), Path(workdir)) as (served, _):
            yield served

    # -- a server ---------------------------------------------------------------------------

    @contextmanager
    def _serve(self, manifest: Path, workdir: Path) -> Iterator[tuple[ServedPolicy, bytes]]:
        """One `icil_policy.serve` process, listening; killed with its group on the way out."""
        workdir.mkdir(parents=True, exist_ok=True)
        # A Unix socket's path is limited to about 100 bytes, which a run directory can exceed.
        sockets = Path(tempfile.mkdtemp(prefix="icil-duel-"))
        address = sockets / SOCKET_FILE
        log_file = workdir / LOG_FILE
        key = secrets.token_bytes(AUTHKEY_BYTES)
        env = {**benchmark_environment(os.environ, AUTHKEY_ENV), AUTHKEY_ENV: key.hex()}
        argv = [
            self.python,
            "-m",
            "icil_policy.serve",
            "--manifest",
            str(manifest),
            "--address",
            str(address),
            "--authkey-env",
            AUTHKEY_ENV,
            "--log-file",
            str(log_file),
        ]
        try:
            process = subprocess.Popen(
                argv,
                env=env,
                cwd=sockets,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            shutil.rmtree(sockets, ignore_errors=True)
            raise RuntimeUnavailable(f"the policy server could not be started: {exc}") from None
        ledger = Ledger(workdir)
        try:
            ledger.started(process.pid)
            self._wait_listening(process, address, log_file)
            served = ServedPolicy(
                address=str(address),
                authkey_env=AUTHKEY_ENV,
                env={AUTHKEY_ENV: key.hex()},
                log_file=log_file,
                died=lambda: self._died(process, log_file),
            )
            self._started(process, served)
            yield served, key
        finally:
            _kill_group(process)
            ledger.ended(process.pid)
            shutil.rmtree(sockets, ignore_errors=True)

    def _wait_listening(self, process: subprocess.Popen, address: Path, log_file: Path) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        while not address.exists():
            code = process.poll()
            if code is not None:
                raise PolicyDied(
                    f"the policy exited ({code}) before listening\n{tail(log_file)}".rstrip()
                )
            if time.monotonic() >= deadline:
                raise PolicyDied(
                    f"the policy did not listen within {self.start_timeout_s:g}s\n"
                    f"{tail(log_file)}".rstrip()
                )
            time.sleep(POLL_S)

    def _started(self, process: subprocess.Popen, served: ServedPolicy) -> None:
        """Called once a server listens. The tests stand in here to kill one."""

    @staticmethod
    def _died(process: subprocess.Popen, log_file: Path) -> PolicyEnd | None:
        """How the server ended underneath its unit: any end but a clean exit after its client.
        It is killed only on the way out of `_serve`, after this is read, so an end seen here is
        the policy's own - on this host nothing tells a policy that exited from one killed by the
        machine's own OOM killer."""
        try:
            code = process.wait(timeout=EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            return None  # still serving a client that never came or never left: not dead
        if code == 0:
            return None
        how = f"was killed by signal {-code}" if code < 0 else f"exited {code}"
        return PolicyEnd(f"the policy process {how}\n{tail(log_file)}".rstrip(), POLICY)


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.kill()
    process.wait()
