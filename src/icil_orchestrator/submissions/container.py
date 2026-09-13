"""A submission's policy, served inside `spec.submission.sandbox`.

    docker run --detach --network none --read-only --tmpfs /tmp --user 1000:1000 --gpus 1
               --memory N --memory-swap N --cpus ... --pids-limit ... --cap-drop ALL
               --security-opt no-new-privileges --mount type=bind,src=<dir>,dst=/run/icil
               --env ICIL_POLICY_AUTHKEY <image>
               python -m icil_policy.serve --manifest /submission/icil.yaml
                      --address /run/icil/policy.sock --authkey-env ICIL_POLICY_AUTHKEY
                      --log-file /run/icil/policy.log

Every limit is the spec's, read here and nowhere else. `--memory-swap` equal to `--memory` means
no swap at all: the spec's bytes are the container's total, where Docker's default would allow as
much again in swap. What the container can reach is its own
image and one directory, shared for the Unix socket and the server's log: mode 0700 on the host
and owned by the sandbox user, so the policy can create the socket and nobody else on the host
can open it. Nothing of the store, the queue, the prompts or the other side is mounted, and the
only variable that crosses is the authkey, by name: `docker run --env NAME` takes the value from
the docker client's environment, so it is never on a command line.

The health check is `hello` through `icil_policy.client.RemotePolicy` within
`budgets.policy_start_seconds`, which builds the policy inside the container. The container is
removed (`docker rm -f`) when the host object closes, however things went.
"""

from __future__ import annotations

import os
import secrets
import stat
import time
from pathlib import Path
from typing import Any

from icil_policy.client import RemotePolicy
from icil_policy.errors import PolicyUnavailable

from .docker import CONTAINER_LABEL, Docker
from .errors import SubmissionError, SubmissionRejected
from .image import SUBMISSION_DIR, sandbox_user

#: The variable the authkey travels in. `benchmarks.check` hands `run_command` the same name.
AUTHKEY_ENV = "ICIL_POLICY_AUTHKEY"
AUTHKEY_BYTES = 32
#: The shared directory as the container sees it, and what the server puts there.
SOCKET_DIR = "/run/icil"
SOCKET_FILE = "policy.sock"
LOG_FILE = "policy.log"
#: How often the start is looked at while nothing listens yet.
POLL_S = 0.1
#: Not in the spec because they are not limits a competitor sees: a non-root user with no
#: capabilities at all, and no way to gain any through a setuid binary in its own image.
HARDENING = ("--cap-drop", "ALL", "--security-opt", "no-new-privileges")


def serve_argv(spec: Any) -> list[str]:
    """What runs in the container: the protocol's server, on the manifest the spec names."""
    manifest = str(spec.submission["manifest"])
    return [
        "python",
        "-m",
        "icil_policy.serve",
        "--manifest",
        f"{SUBMISSION_DIR}/{manifest}",
        "--address",
        f"{SOCKET_DIR}/{SOCKET_FILE}",
        "--authkey-env",
        AUTHKEY_ENV,
        "--log-file",
        f"{SOCKET_DIR}/{LOG_FILE}",
    ]


def run_argv(
    spec: Any, *, image: str, name: str, socket_dir: Path, gpus: int | None = None
) -> list[str]:
    """`docker run`'s arguments for `image` under the spec's sandbox, `run` itself excluded.

    `gpus` overrides `sandbox.gpus` for a policy that needs none (the examples, the tests); the
    spec's count is the default.
    """
    sandbox = spec.submission["sandbox"]
    args = [
        "--detach",
        "--name",
        name,
        "--label",
        CONTAINER_LABEL,
        "--network",
        str(sandbox["network"]),
    ]
    if sandbox["read_only_root"]:
        args.append("--read-only")
    for path in sandbox["tmpfs"]:
        args += ["--tmpfs", str(path)]
    args += ["--user", str(sandbox["user"])]
    count = int(sandbox["gpus"]) if gpus is None else int(gpus)
    if count > 0:
        args += ["--gpus", str(count)]
    memory = str(int(sandbox["memory_bytes"]))
    args += [
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--cpus",
        str(sandbox["cpus"]),
        "--pids-limit",
        str(int(sandbox["pids"])),
        *HARDENING,
        "--mount",
        f"type=bind,src={socket_dir},dst={SOCKET_DIR}",
        "--env",
        AUTHKEY_ENV,
        image,
        *serve_argv(spec),
    ]
    return args


def is_socket(path: Path) -> bool:
    """True when `path` is a Unix socket itself - not a symbolic link to one, or to anything, and
    not a file of another kind. The policy can write to the socket's directory, so what is there
    is looked at without following it."""
    try:
        return stat.S_ISSOCK(os.lstat(path).st_mode)
    except OSError:
        return False


def prepare_socket_dir(directory: Path, spec: Any) -> Path:
    """The shared directory: created, mode 0700, owned by the sandbox user."""
    uid, gid = sandbox_user(spec)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    try:
        os.chown(directory, uid, gid)
    except PermissionError as exc:
        raise SubmissionError(
            f"cannot give {directory} to uid {uid} (the sandbox user), which must own it to "
            f"create the socket: {exc}"
        ) from None
    for stale in (SOCKET_FILE, LOG_FILE):
        (directory / stale).unlink(missing_ok=True)
    return directory


class PolicyContainer:
    """One served policy in one container. `with PolicyContainer(...) as c: c.hello(60)`."""

    def __init__(
        self,
        spec: Any,
        docker: Docker,
        image: str,
        *,
        name: str,
        socket_dir: Path,
        gpus: int | None = None,
    ) -> None:
        self.spec = spec
        self.docker = docker
        self.image = image
        self.name = name
        self.socket_dir = Path(socket_dir)
        self.gpus = gpus
        self.authkey = secrets.token_bytes(AUTHKEY_BYTES)
        self.session: RemotePolicy | None = None
        self.started_at: float | None = None
        self.listening_after_s: float | None = None
        self._closed = False
        self._ran = False

    @property
    def socket_path(self) -> Path:
        return self.socket_dir / SOCKET_FILE

    @property
    def log_path(self) -> Path:
        return self.socket_dir / LOG_FILE

    def start(self) -> None:
        prepare_socket_dir(self.socket_dir, self.spec)
        args = run_argv(
            self.spec, image=self.image, name=self.name, socket_dir=self.socket_dir, gpus=self.gpus
        )
        self._ran = True
        self.started_at = time.monotonic()
        self.docker.run(args, env={AUTHKEY_ENV: self.authkey.hex()})

    def hello(self, timeout_s: float) -> dict[str, Any]:
        """Wait for the server to listen, then `hello`, all within `timeout_s`; the reply.

        The session stays open on `self.session`: the server serves one client, so whoever drives
        the policy next uses this one. A policy that cannot be built, a server that never listens
        or a container that exits first is a rejection with the reason and the log's tail.
        """
        if self.started_at is None:
            self.start()
        deadline = self.started_at + float(timeout_s)
        self._wait_listening(deadline)
        remaining = max(deadline - time.monotonic(), POLL_S)
        try:
            policy = RemotePolicy(
                str(self.socket_path), self.authkey, timeout_s=remaining, log_file=self.log_path
            )
            reply = policy.hello()
        except PolicyUnavailable as exc:
            raise SubmissionRejected("hello", str(exc)) from None
        self.session = policy
        return reply

    def _wait_listening(self, deadline: float) -> None:
        while True:
            if is_socket(self.socket_path):
                self.listening_after_s = round(time.monotonic() - self.started_at, 3)  # type: ignore[operator]
                return
            state = self.docker.state(self.name)
            if not state.running:
                code = state.exit_code
                raise SubmissionRejected(
                    "start",
                    f"the policy container exited ({code}) before listening"
                    f"{': ' + state.error if state.error else ''}\n{self._log_tail()}",
                )
            if time.monotonic() >= deadline:
                raise SubmissionRejected(
                    "start",
                    f"the policy did not listen within policy_start_seconds "
                    f"({deadline - self.started_at:g}s)\n{self._log_tail()}",  # type: ignore[operator]
                )
            time.sleep(POLL_S)

    def _log_tail(self) -> str:
        from icil_policy.logs import tail

        served = tail(self.log_path)
        return served if served else self.docker.logs(self.name)

    def exec(self, argv: list[str], *, timeout_s: float = 60):
        """Run `argv` inside the container as the sandbox user (the tests look around with it)."""
        return self.docker.exec(self.name, argv, timeout_s=timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.session is not None:
            self.session.close()
            self.session = None
        if self._ran:
            self.docker.remove(self.name)

    def __enter__(self) -> PolicyContainer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - a finaliser raises to nobody
            pass
