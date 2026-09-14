"""A submission: a Hugging Face repository at a commit, fetched, checked, built and run sandboxed.

The pieces, in the order a submission goes through them:

- `resolve`: `repo@revision` to the 40-hex commit sha it names, through the Hub, once, at queue
  time. Everything published hangs off that sha.
- `fetch`: the repository at that sha into a content-addressed cache, no larger than
  `spec.submission.max_repo_bytes`.
- `checks`: `icil.yaml` and what it names, looked at without importing anything from it.
- `image`: the pinned base image and a submission's own image, built from it by digest.
- `container`: the served policy inside `spec.submission.sandbox`, spoken to over one socket.
- `check`: all of the above in order, for `icil-orchestrator submission check`.

Nothing here imports, unpickles or executes a submission's code: the orchestrator reads its
manifest, builds its image and talks to its container.
"""

from .errors import SubmissionError, SubmissionRejected

__all__ = ["SubmissionError", "SubmissionRejected"]
