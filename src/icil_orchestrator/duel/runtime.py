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
  whatever a policy remembers from leaking from one unit into the next. When the policy never
  listens, `serve` raises `PolicyDied`; after the unit, `ServedPolicy.died()` says whether it died
  underneath the benchmark. Either way the unit is void, and so are the side's remaining units.

Two runtimes implement it: `docker_runtime.DockerPolicyRuntime`, the sandbox (#4's containers),
and `local_runtime.SubprocessPolicyRuntime`, which runs a local directory's policy on this host
with no sandbox at all, for development and the tests.
"""

from __future__ import annotations

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
    """A served policy never listened, or ended while its unit was being played."""


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


def _alive() -> str | None:
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
    #: Called once the unit is over: why the policy died underneath it, or None if it did not.
    died: Callable[[], str | None] = _alive


@runtime_checkable
class PolicyRuntime(Protocol):
    #: `docker` or `local`, recorded with every duel it serves.
    name: str

    def resolve(self, repo: str, revision: str) -> SubmissionRef: ...

    def fetch(self, ref: SubmissionRef, *, workdir: Path) -> FetchedSubmission: ...

    def prepare(self, fetched: FetchedSubmission, *, workdir: Path) -> PreparedSubmission: ...

    def serve(
        self, prepared: PreparedSubmission, *, workdir: Path
    ) -> AbstractContextManager[ServedPolicy]: ...
