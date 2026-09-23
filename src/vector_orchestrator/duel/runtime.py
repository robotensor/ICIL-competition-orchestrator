"""Where a submission's policy runs, as a duel sees it: one small seam.

A duel never builds an image or starts a container itself. It asks a `PolicyRuntime` three
things, in order:

- `resolve(repo, revision)`: the `SubmissionRef` at the commit that will run.
- `fetch(ref)`, then `prepare(fetched)`: once per side, before a unit is played. Fetching is the
  `fetching` phase and preparing (manifest, image, a health check) is `checking`, which is why
  they are two calls. A submission that cannot run raises `SubmissionRefused` with a reason that
  can be published; a harness that cannot do its part raises `RuntimeUnavailable`, which judges
  nobody.
- `serve(prepared, workdir=...)`: a context manager that serves the policy **for one unit** and
  yields where to reach it. `python -m vector_policy.serve` accepts exactly one client and exits
  when it hangs up, so every unit gets a fresh policy process - which is also what keeps
  whatever a policy remembers from leaking from one unit into the next, and why a policy that died
  on one unit is simply served again for the next.

Whose a failure is decides what it costs, so the seam keeps the two apart. A policy that never
listens within `budgets.policy_start_seconds`, or exits before it does, raises `PolicyDied`: the
submission's own doing, that side's failure on the unit. A runtime that cannot serve at all (no
Docker, a container removed from outside) raises `RuntimeUnavailable`: nobody's doing, void for
both. After the unit, `ServedPolicy.died()` says how the policy ended if it did not end cleanly, as
a `PolicyEnd` whose `cause` is one or the other.

Two runtimes implement it: `docker_runtime.DockerPolicyRuntime`, the sandbox (#4's containers),
and `local_runtime.SubprocessPolicyRuntime`, which runs a local directory's policy on this host
with no sandbox at all, for development and the tests.
"""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..ids import SubmissionRef


class SubmissionRefused(Exception):
    """The submission cannot run, for a reason of its own; `step` says where it stopped."""

    def __init__(self, step: str, reason: str) -> None:
        self.step = step
        self.reason = reason
        super().__init__(f"{step}: {reason}")


class RuntimeUnavailable(RuntimeError):
    """The runtime could not take a submission through a step (no Docker, no Hub); nobody's loss."""


class PolicyDied(RuntimeError):
    """A served policy never listened: it exited first, or did not listen in time. Its side's
    failure on the unit, never a void."""


#: Who brought about a unit's end: the side's own submission, or the harness around it. The same
#: two words a benchmark's `read_result` may give as `void_cause`.
POLICY = "policy"
HARNESS = "harness"
CAUSES = (POLICY, HARNESS)

#: How much of a policy's log the copy a benchmark reads holds: its end, which is all a failure
#: quotes (`vector_policy.logs.tail` reads at most 8 KiB).
LOG_TAIL_BYTES = 1 << 16
#: How often that copy catches up with the log.
LOG_MIRROR_S = 0.05

#: A policy can write where its log is, so it may have made the log a link or a pipe: the log is
#: opened without following a link or waiting on a pipe, and read only if it is a regular file.
_LOG_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


@dataclass(frozen=True)
class PolicyEnd:
    """How a unit's policy ended underneath it, when it did not end cleanly.

    `cause` is `policy` for an end its submission can bring about - a non-zero exit, a signal,
    running out of the sandbox's memory - and `harness` for one it cannot: its container removed
    from outside, or Docker not answering about it.
    """

    reason: str
    cause: str


@dataclass(frozen=True)
class FetchedSubmission:
    """A submission's code, at the commit it resolved to, where the runtime can build it."""

    ref: SubmissionRef
    #: The commit the code is at: the Hub's sha, or a local directory's tree hash.
    commit: str
    root: Path
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedSubmission:
    """A submission ready to serve: checked, built, and answering `hello`."""

    ref: SubmissionRef
    commit: str
    #: The base image the submission's image was built from, by digest; None where there is none.
    base_image_digest: str | None
    #: The submission's own image id; None for a runtime that builds none.
    image: str | None
    #: `module:Class` from its `icil.yaml`, and what it answered `hello` with.
    policy: str | None
    action_type: str | None
    #: Whatever the runtime needs to serve it again (an image tag, a checkout); not published.
    handle: Any = field(default=None, compare=False, repr=False)

    def as_side(self) -> dict[str, Any]:
        """What an event record carries for this submission as one side of a duel."""
        return {
            "commit": self.commit,
            "base_image_digest": self.base_image_digest,
            "image": self.image,
            "policy": self.policy,
            "action_type": self.action_type,
        }


def copy_log(source: Path, target: Path, limit: int) -> None:
    """Append at most `limit` bytes of a policy's log to `target`. The policy can write where its
    log is, so it may have made it a link or a pipe: only a regular file is read, and nothing is
    followed or waited on."""
    try:
        fd = os.open(source, _LOG_FLAGS)
    except OSError:
        return
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return
        with open(fd, "rb", closefd=False) as fh, open(target, "ab") as out:
            out.write(fh.read(limit))
    except OSError:
        pass
    finally:
        os.close(fd)


def _copy_tail(source: Path, target: Path, limit: int, seen: Any) -> Any:
    """Replace `target` with the last `limit` bytes of the log at `source`, unless the log is as it
    was when `seen` was returned; what to pass as `seen` next time. A log that is not a regular
    file, or cannot be read, leaves `target` as it was."""
    try:
        fd = os.open(source, _LOG_FLAGS)
    except OSError:
        return seen
    try:
        info = os.fstat(fd)
        state = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if not stat.S_ISREG(info.st_mode) or state == seen:
            return seen
        with open(fd, "rb", closefd=False) as fh:
            fh.seek(max(0, info.st_size - limit))
            data = fh.read(limit)
        partial = target.with_name(f".{target.name}.partial")
        partial.write_bytes(data)
        os.replace(partial, target)  # a reader never sees half of a copy
        return state
    except OSError:
        return seen
    finally:
        os.close(fd)


@contextmanager
def mirror_log(
    source: Path,
    target: Path,
    *,
    limit: int = LOG_TAIL_BYTES,
    interval_s: float = LOG_MIRROR_S,
) -> Iterator[Path]:
    """`target`, holding the end of the policy's log at `source` for as long as the block runs.

    A benchmark quotes a policy's log when the policy fails it, and must not be handed the log
    itself: the policy can write where its log is, so it could swap the file for a link to
    something on the benchmark's side, or a pipe that blocks whoever opens it. `target` is a file
    in a directory the policy never sees, replaced whole with the last `limit` bytes of the log
    whenever the log changes, read only from a regular file without following a link; it trails
    the log by up to `interval_s`, catches up once more on the way out, and is removed then."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    seen: Any = _copy_tail(source, target, limit, None)
    stop = threading.Event()

    def follow() -> None:
        nonlocal seen
        while not stop.wait(interval_s):
            seen = _copy_tail(source, target, limit, seen)

    thread = threading.Thread(target=follow, name="vector-policy-log", daemon=True)
    thread.start()
    try:
        yield target
    finally:
        stop.set()
        thread.join()
        target.unlink(missing_ok=True)


def _alive() -> PolicyEnd | None:
    return None


@dataclass(frozen=True)
class ServedPolicy:
    """One unit's policy, listening.

    The benchmark subprocess gets `address` and the *name* `authkey_env`; `env` is what must be
    added to its environment for that name to hold the key. `log_file` is the server's log (and
    everything the policy printed), kept in the unit's directory once the unit is over; `live_log`
    is that log as the policy writes it while it serves, which only `mirror_log` reads.
    """

    address: str
    authkey_env: str
    env: dict[str, str]
    log_file: Path
    #: Called once the unit is over: how the policy ended underneath it, or None if it did not.
    died: Callable[[], PolicyEnd | None] = _alive
    #: The log the policy writes while it serves; None where the runtime has none.
    live_log: Path | None = None
    #: The policy server's process group on this host, whose CPU counts as the unit's progress for
    #: the stall watchdog (a benchmark waiting on a policy that is building its model is not
    #: hung); None where the runtime cannot say (a container).
    pgid: int | None = None


@runtime_checkable
class PolicyRuntime(Protocol):
    #: `docker` or `local`, recorded with every duel it serves.
    name: str

    def reap(self) -> list[str]:
        """Remove what a killed process started and could not tear down; the names of what was
        removed. Called by the orchestrator on start, holding the store's lock."""
        ...

    def resolve(self, repo: str, revision: str) -> SubmissionRef: ...

    def fetch(self, ref: SubmissionRef, *, workdir: Path) -> FetchedSubmission: ...

    def prepare(self, fetched: FetchedSubmission, *, workdir: Path) -> PreparedSubmission: ...

    def serve(
        self, prepared: PreparedSubmission, *, workdir: Path
    ) -> AbstractContextManager[ServedPolicy]: ...
