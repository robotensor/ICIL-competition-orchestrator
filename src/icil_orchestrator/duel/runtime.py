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
  yields where to reach it. `python -m icil_policy.serve` accepts exactly one client and exits
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
from collections.abc import Callable
from contextlib import AbstractContextManager
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
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(source, flags)
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


def _alive() -> PolicyEnd | None:
    return None


@dataclass(frozen=True)
class ServedPolicy:
    """One unit's policy, listening.

    The benchmark subprocess gets `address` and the *name* `authkey_env`; `env` is what must be
    added to its environment for that name to hold the key. `log_file` is the server's log (and
    everything the policy printed), kept in the unit's directory.
    """

    address: str
    authkey_env: str
    env: dict[str, str]
    log_file: Path
    #: Called once the unit is over: how the policy ended underneath it, or None if it did not.
    died: Callable[[], PolicyEnd | None] = _alive


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
