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
much again in swap. What the container can reach is its own image and one directory, shared for
the Unix socket and the server's log: mode 0700 on the host and owned by the sandbox user, so the
policy can create the socket and nobody else on the host can open it. The policy can write there,
so the directory is a tmpfs of `SHARED_DIR_BYTES` mounted on the host by the orchestrator (root)
for the container's lifetime: a policy that fills it fills nothing else, and the log is copied out
before the tmpfs goes. Nothing of the store, the queue, the prompts or the other side is mounted,
and the only variable that crosses is the authkey, by name: `docker run --env NAME` takes the
value from the docker client's environment, so it is never on a command line.

The health check is `hello` through `icil_policy.client.RemotePolicy` within
`budgets.policy_start_seconds`, which builds the policy inside the container. The container is
removed (`docker rm -f`) when the host object closes, however things went.
"""

from __future__ import annotations

import os
import secrets
import shutil
import stat
import subprocess
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
#: The shared directory's tmpfs: its size and how many entries it takes. The socket and the log
#: fit; a policy that writes more gets ENOSPC, and the host's disk sees none of it. What a
#: competitor sees is a log that stops growing, so this is not a spec limit either.
SHARED_DIR_BYTES = 64 << 20
SHARED_DIR_INODES = 64
#: What the tmpfs is listed under in the host's mount table.
SHARED_DIR_SOURCE = "icil-policy"


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


def can_bound_shared_dir() -> bool:
    """Whether this process can mount the shared directory's tmpfs: root, with `mount` at hand."""
    return (
        os.geteuid() == 0
        and shutil.which("mount") is not None
        and shutil.which("umount") is not None
    )


def mount_shared_dir(directory: Path, uid: int, gid: int) -> None:
    """A tmpfs of `SHARED_DIR_BYTES` at `directory`, owned by the sandbox user, mode 0700."""
    options = f"size={SHARED_DIR_BYTES},nr_inodes={SHARED_DIR_INODES},uid={uid},gid={gid},mode=0700"
    _mount_command(["mount", "-t", "tmpfs", "-o", options, SHARED_DIR_SOURCE, str(directory)])


def unmount_shared_dir(directory: Path) -> None:
    """The tmpfs off `directory`; lazily when something still holds it, so it goes when that
    ends."""
    try:
        _mount_command(["umount", str(directory)])
    except SubmissionError:
        _mount_command(["umount", "--lazy", str(directory)])


def _mount_command(argv: list[str]) -> None:
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SubmissionError(f"{' '.join(argv)} failed: {exc}") from None
    if done.returncode != 0:
        raise SubmissionError(f"{' '.join(argv)} failed ({done.returncode}): {done.stderr.strip()}")


def is_shared_mount(directory: Path) -> bool:
    """Whether the host's mount table lists our tmpfs at `directory` - a run that never got to
    release it, or one still running."""
    try:
        table = Path("/proc/self/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    wanted = os.path.realpath(directory)
    for line in table.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == SHARED_DIR_SOURCE and fields[2] == "tmpfs":
            # The kernel escapes a space in a path as \040 and so on.
            mounted = fields[1].encode("utf-8").decode("unicode_escape")
            if mounted == wanted:
                return True
    return False


def prepare_socket_dir(directory: Path, spec: Any, *, bounded: bool | None = None) -> bool:
    """The shared directory: created, mode 0700, owned by the sandbox user, and a tmpfs of
    `SHARED_DIR_BYTES` when `bounded` (by default, when this process can mount one). Whether
    it is bounded is returned; a plain directory has no cap on what the policy writes there.
    """
    uid, gid = sandbox_user(spec)
    if bounded is None:
        bounded = can_bound_shared_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if is_shared_mount(directory):
        unmount_shared_dir(directory)  # left by a run that did not get to release it
    if bounded:
        mount_shared_dir(directory, uid, gid)
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
    return bounded


def release_socket_dir(directory: Path) -> None:
    """The tmpfs off the shared directory, with the log kept: it is read out first and written
    to the plain directory underneath, so `--work` keeps what the server said."""
    kept = _read_plain_file(directory / LOG_FILE, SHARED_DIR_BYTES)
    unmount_shared_dir(directory)
    if kept is not None:
        (directory / LOG_FILE).write_bytes(kept)


def _read_plain_file(path: Path, limit: int) -> bytes | None:
    """Up to `limit` bytes of the regular file at `path`; None for anything else there. The policy
    could have replaced the log with a link or a pipe, and neither is followed or waited on."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with open(fd, "rb", closefd=False) as handle:
            return handle.read(limit)
    except OSError:
        return None
    finally:
        os.close(fd)


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
        bounded: bool | None = None,
    ) -> None:
        self.spec = spec
        self.docker = docker
        self.image = image
        self.name = name
        self.socket_dir = Path(socket_dir)
        self.gpus = gpus
        #: Whether the shared directory is a bounded tmpfs; None until started, and by default
        #: whatever this process can do (`can_bound_shared_dir`).
        self.bounded: bool | None = bounded
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
        self.bounded = prepare_socket_dir(self.socket_dir, self.spec, bounded=self.bounded)
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
        if self.bounded:
            try:
                release_socket_dir(self.socket_dir)
            except SubmissionError:
                pass  # at most SHARED_DIR_BYTES stay mounted; the next start unmounts them

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
