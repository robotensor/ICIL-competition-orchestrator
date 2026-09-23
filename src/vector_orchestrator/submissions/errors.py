"""The two ways a submission fails to run, kept apart because they mean different things.

A rejection is the submission's own doing and its reason is publishable: the repository is not
there, is too large, its manifest is wrong, its requirements do not install, its policy does not
build. A `SubmissionError` is the harness's: the Hub is unreachable, Docker is not there, the base
image is not on this host. The first is a verdict on the entry; the second is nobody's loss.
"""

from __future__ import annotations


class SubmissionRejected(ValueError):
    """The submission cannot run, for a reason that is its own. `step` names where it failed."""

    def __init__(self, step: str, reason: str) -> None:
        self.step = step
        self.reason = reason
        super().__init__(f"{step}: {reason}")


class SubmissionError(RuntimeError):
    """The harness could not take the submission through a step; the submission is not judged."""
