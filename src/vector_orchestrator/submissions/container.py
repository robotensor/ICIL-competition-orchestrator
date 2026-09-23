"""A submission's policy, served inside `spec.submission.sandbox`.

    docker run --detach --network none --read-only --tmpfs /tmp:exec,nosuid,nodev,size=N
               --user 1000:1000 --gpus 1 --memory N --memory-swap N --cpus ... --pids-limit ...
               --cap-drop ALL --security-opt no-new-privileges
               --mount type=bind,src=<dir>,dst=/run/vector
               --env HOME=/tmp/home --env TMPDIR=/tmp --env XDG_CACHE_HOME=/tmp/home/.cache
               --env TRITON_CACHE_DIR=... --env TORCHINDUCTOR_CACHE_DIR=...
               --env TORCH_EXTENSIONS_DIR=... --env VECTOR_POLICY_AUTHKEY <image>
               sh -c 'mkdir -p -m 0700 "$HOME" "$XDG_CACHE_HOME" && exec "$@"' sh
               python -m vector_policy.serve --manifest /submission/icil.yaml
                      --address /run/vector/policy.sock --authkey-env VECTOR_POLICY_AUTHKEY
                      --log-file /run/vector/policy.log

Every limit is the spec's, read here and nowhere else. `--memory-swap` equal to `--memory` means
no swap at all: the spec's bytes are the container's total, where Docker's default would allow as
much again in swap.

**The scratch tmpfs.** Each `sandbox.tmpfs` path is a tmpfs of `tmpfs_bytes`, `nosuid,nodev`
always, and `exec` when `tmpfs_exec` says so, so a policy can JIT-compile: torch.compile, Triton,
cffi and `torch.utils.cpp_extension` write a shared object and load it. The first path is the
scratch directory: `TMPDIR`, the policy's `HOME` (made by the shell above, since the tmpfs is
empty at every start) and the cache root every compiler is pointed at (`policy_environment`).
Submission code already runs natively in here, so code it compiles adds no reach; what bounds it
is the network, the read-only root, the non-root user and the limits, and the size cap keeps the
scratch space inside `memory_bytes`, which its pages are charged to.

What the container can reach is its own image and one directory, shared for
the Unix socket and the server's log: mode 0700 on the host and owned by the sandbox user, so the
policy can create the socket and nobody else on the host can open it. The policy can write there,
so the directory is a tmpfs of `SHARED_DIR_BYTES` mounted on the host by the orchestrator (root)
for the container's lifetime, nosuid, nodev and noexec: a policy that fills it fills nothing else,
nothing it writes there runs, and the log is copied out before the tmpfs goes. Nothing of the store, the queue, the prompts or the other side is mounted,
and the only variable that crosses is the authkey, by name: `docker run --env NAME` takes the
value from the docker client's environment, so it is never on a command line.

The health check is `hello` through `vector_policy.client.RemotePolicy` within
`budgets.policy_start_seconds`, which builds the policy inside the container. The container is
removed (`docker rm -f`) when the host object closes, however things went.

A process killed before it closes leaves its container behind, and one killed between `docker
run` and `hello` leaves a server that waits for a client for ever. So every container is labelled
with the process that started it (`Owner`: its pid, that process's start time, since a pid is
reused, and the pid namespace both are in), and each start first reaps the containers whose owner
is gone (`reap_orphans`), with their shared tmpfs. Whether a client ever connected is not asked:
the socket's directory is the policy's to write in, so nothing there is evidence.
"""

from __future__ import annotations

import os
import secrets
import shutil
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from vector_policy.client import RemotePolicy
from vector_policy.errors import PolicyUnavailable

from .docker import CONTAINER_LABEL, Docker
from .errors import SubmissionError, SubmissionRejected
from .image import SUBMISSION_DIR, sandbox_user

#: The variable the authkey travels in. `benchmarks.check` hands `run_command` the same name.
AUTHKEY_ENV = "VECTOR_POLICY_AUTHKEY"
AUTHKEY_BYTES = 32
#: The shared directory as the container sees it, and what the server puts there.
SOCKET_DIR = "/run/vector"
SOCKET_FILE = "policy.sock"
LOG_FILE = "policy.log"
#: How often the start is looked at while nothing listens yet.
POLL_S = 0.1
#: Not in the spec because they are not limits a competitor sees: a non-root user with no
#: capabilities at all, and no way to gain any through a setuid binary in its own image.
HARDENING = ("--cap-drop", "ALL", "--security-opt", "no-new-privileges")
#: Options every `sandbox.tmpfs` mount gets whatever the spec says: a setuid bit or a device file
#: written there means nothing. Whether code written there may run (`tmpfs_exec`) and how much
#: fits (`tmpfs_bytes`) are the spec's.
TMPFS_HARDENING = ("nosuid", "nodev")
#: Under the scratch directory (the first `sandbox.tmpfs` path): the policy's home, and in it the
#: cache root.
HOME_SUBDIR = "home"
CACHE_SUBDIR = ".cache"
#: Each JIT compiler's cache, as the variable that moves it and its directory under the cache root.
#: Their defaults are scattered (~/.triton/cache, /tmp/torchinductor_<user>,
#: ~/.cache/torch_extensions); named, they are in one place a policy can find, whatever a library
#: defaults to next.
CACHE_ENV = (
    ("TRITON_CACHE_DIR", "triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
)
#: What runs before the server: the home and the cache root made on the empty tmpfs, then the
#: server in the shell's place, so it is still the container's first process.
PREPARE_HOME = 'mkdir -p -m 0700 "$HOME" "$XDG_CACHE_HOME" && exec "$@"'
#: The shared directory's tmpfs: its size and how many entries it takes. The socket and the log
#: fit; a policy that writes more gets ENOSPC, and the host's disk sees none of it. What a
#: competitor sees is a log that stops growing, so this is not a spec limit either.
SHARED_DIR_BYTES = 64 << 20
SHARED_DIR_INODES = 64
#: What the tmpfs is listed under in the host's mount table.
SHARED_DIR_SOURCE = "vector-policy"
#: The shared tmpfs holds a socket and a log, so nothing written there runs, and a setuid bit or
#: a device file means nothing; the bind mount into the container keeps these flags. The spec's
#: tmpfs is then the only place code a policy writes can run from.
SHARED_DIR_HARDENING = ("nosuid", "nodev", "noexec")
#: Labels on every container: the process that started it (`Owner`) and its shared directory on
#: the host, so a container whose owner is gone can be found and reaped with its tmpfs.
OWNER_PID_LABEL = "vector.owner.pid"
OWNER_START_LABEL = "vector.owner.start"
OWNER_PIDNS_LABEL = "vector.owner.pidns"
SHARED_DIR_LABEL = "vector.shared-dir"


@dataclass(frozen=True)
class Owner:
    """A process as the kernel names it for as long as it lives: its pid, its start time in clock
    ticks since boot (field 22 of /proc/<pid>/stat; a later process given the same pid starts
    later) and its pid namespace (a pid means something only in its own)."""

    pid: int
    start: str
    pidns: str

    @classmethod
    def current(cls) -> Owner | None:
        """This process; None where /proc cannot say (not Linux), and then nothing is reaped."""
        try:
            return cls(os.getpid(), _start_ticks("self"), os.readlink("/proc/self/ns/pid"))
        except (OSError, ValueError, IndexError):
            return None

    @classmethod
    def from_labels(cls, labels: Mapping[str, str]) -> Owner | None:
        try:
            return cls(
                int(labels[OWNER_PID_LABEL]), labels[OWNER_START_LABEL], labels[OWNER_PIDNS_LABEL]
            )
        except (KeyError, ValueError):
            return None

    def labels(self) -> dict[str, str]:
        return {
            OWNER_PID_LABEL: str(self.pid),
            OWNER_START_LABEL: self.start,
            OWNER_PIDNS_LABEL: self.pidns,
        }

    def gone(self) -> bool:
        """Whether this process has ended, as seen from its own pid namespace. Anything /proc does
        not answer plainly (no permission, an unreadable line) is taken as not gone."""
        try:
            return _start_ticks(str(self.pid)) != self.start
        except FileNotFoundError:
            return True
        except (OSError, ValueError, IndexError):
            return False


def _start_ticks(pid: str) -> str:
    stat_line = Path(f"/proc/{pid}/stat").read_text()
    # The command name is in parentheses and may hold anything, spaces and parentheses included;
    # the fields after its last ")" start at field 3, so field 22 is the 20th of them.
    return stat_line.rpartition(")")[2].split()[19]


def serve_argv(spec: Any) -> list[str]:
    """What runs in the container: the protocol's server, on the manifest the spec names."""
    manifest = str(spec.submission["manifest"])
    return [
        "python",
        "-m",
        "vector_policy.serve",
        "--manifest",
        f"{SUBMISSION_DIR}/{manifest}",
        "--address",
        f"{SOCKET_DIR}/{SOCKET_FILE}",
        "--authkey-env",
        AUTHKEY_ENV,
        "--log-file",
        f"{SOCKET_DIR}/{LOG_FILE}",
    ]


def tmpfs_options(spec: Any) -> str:
    """The options every `sandbox.tmpfs` mount takes: exec or not, `TMPFS_HARDENING`, the size."""
    sandbox = spec.submission["sandbox"]
    runs_code = "exec" if sandbox["tmpfs_exec"] else "noexec"
    return ",".join((runs_code, *TMPFS_HARDENING, f"size={int(sandbox['tmpfs_bytes'])}"))


def scratch_dir(spec: Any) -> PurePosixPath:
    """The first `sandbox.tmpfs` path: where the policy's home and its caches live."""
    return PurePosixPath(str(spec.submission["sandbox"]["tmpfs"][0]))


def policy_environment(spec: Any) -> dict[str, str]:
    """The variables the container is given besides the authkey: the scratch directory as
    `TMPDIR`, the home in it and every JIT compiler's cache under the home's cache root."""
    scratch = scratch_dir(spec)
    home = scratch / HOME_SUBDIR
    cache = home / CACHE_SUBDIR
    return {
        "HOME": str(home),
        "TMPDIR": str(scratch),
        "XDG_CACHE_HOME": str(cache),
        **{name: str(cache / sub) for name, sub in CACHE_ENV},
    }


def container_argv(spec: Any) -> list[str]:
    """The container's command: `PREPARE_HOME` in a shell, which then becomes the server."""
    return ["sh", "-c", PREPARE_HOME, "sh", *serve_argv(spec)]


def run_argv(
    spec: Any,
    *,
    image: str,
    name: str,
    socket_dir: Path,
    gpus: int | None = None,
    owner: Owner | None = None,
) -> list[str]:
    """`docker run`'s arguments for `image` under the spec's sandbox, `run` itself excluded.

    `gpus` overrides `sandbox.gpus` for a policy that needs none (the examples, the tests); the
    spec's count is the default. `owner`, the process that will close the container, is put on it
    as labels, with the shared directory, for `reap_orphans`.
    """
    sandbox = spec.submission["sandbox"]
    args = ["--detach", "--name", name, "--label", CONTAINER_LABEL]
    labels = {**(owner.labels() if owner is not None else {}), SHARED_DIR_LABEL: str(socket_dir)}
    for key, value in labels.items():
        args += ["--label", f"{key}={value}"]
    args += ["--network", str(sandbox["network"])]
    if sandbox["read_only_root"]:
        args.append("--read-only")
    options = tmpfs_options(spec)
    for path in sandbox["tmpfs"]:
        args += ["--tmpfs", f"{path}:{options}"]
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
    ]
    for key, value in policy_environment(spec).items():
        args += ["--env", f"{key}={value}"]
    args += ["--env", AUTHKEY_ENV, image, *container_argv(spec)]
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
    """A tmpfs of `SHARED_DIR_BYTES` at `directory`, owned by the sandbox user, mode 0700, with
    `SHARED_DIR_HARDENING`."""
    options = ",".join(
        (
            f"size={SHARED_DIR_BYTES},nr_inodes={SHARED_DIR_INODES},uid={uid},gid={gid},mode=0700",
            *SHARED_DIR_HARDENING,
        )
    )
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


def reap_orphans(docker: Docker, *, current: Owner | None = None) -> list[str]:
    """Remove every policy container whose owner is gone, and release its shared tmpfs; the names
    removed. A container with no owner labels, or one started from another pid namespace, is left
    alone: whether its owner lives cannot be told from here. Best effort: a docker that cannot
    list its containers reaps nothing, and says nothing, since the start that follows will."""
    me = Owner.current() if current is None else current
    if me is None:
        return []
    try:
        listed = docker.policy_containers()
    except SubmissionError:
        return []
    reaped = []
    for found in listed:
        owner = Owner.from_labels(found.labels)
        if owner is None or owner.pidns != me.pidns or owner == me or not owner.gone():
            continue
        docker.remove(found.name)
        shared = found.labels.get(SHARED_DIR_LABEL)
        if shared and is_shared_mount(Path(shared)):
            try:
                release_socket_dir(Path(shared))
            except SubmissionError:
                pass  # the next start at that directory unmounts it
        reaped.append(found.name)
    return reaped


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
        #: The containers of processes that are gone, removed by `start` before this one ran.
        self.reaped: list[str] = []
        self._closed = False
        self._ran = False

    @property
    def socket_path(self) -> Path:
        return self.socket_dir / SOCKET_FILE

    @property
    def log_path(self) -> Path:
        return self.socket_dir / LOG_FILE

    def start(self) -> None:
        owner = Owner.current()
        self.reaped = reap_orphans(self.docker, current=owner)
        self.bounded = prepare_socket_dir(self.socket_dir, self.spec, bounded=self.bounded)
        args = run_argv(
            self.spec,
            image=self.image,
            name=self.name,
            socket_dir=self.socket_dir.absolute(),
            gpus=self.gpus,
            owner=owner,
        )
        self._ran = True
        self.started_at = time.monotonic()
        self.docker.run(args, env={AUTHKEY_ENV: self.authkey.hex()})

    def hello(self, timeout_s: float) -> dict[str, Any]:
        """Wait for the server to listen, then `hello`, all within `timeout_s`; the reply.

        The session stays open on `self.session`: the server serves one client, so whoever drives
        the policy next uses this one, and every call on it is bounded by `budgets.act_timeout_s`
        - the start budget was for `hello` alone. A policy that cannot be built, a server that
        never listens or a container that exits first is a rejection with the reason and the
        log's tail.
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
        policy.timeout_s = float(self.spec.budgets["act_timeout_s"])
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
        from vector_policy.logs import tail

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
