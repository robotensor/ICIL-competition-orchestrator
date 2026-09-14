"""`repo@revision` to the commit it names, through the Hugging Face Hub, once.

A branch or a tag is code that can change after it was queued; the queue, the duel id and the
published record all hang off a 40-hex commit sha (`ids.is_commit_sha`), so a name is resolved
here, at queue time, and never looked at again. A sha given as such is confirmed to exist. The
same call lists the repository's files with their sizes, so a repository larger than
`spec.submission.max_repo_bytes` is refused before a byte of it is downloaded.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..ids import SubmissionRef, is_commit_sha, is_repo
from .errors import SubmissionError, SubmissionRejected

#: What a submission is on the Hub. Datasets and Spaces are not submissions.
REPO_TYPE = "model"


@dataclass(frozen=True)
class RepoFile:
    path: str
    #: Bytes, when the Hub says; None for an entry it gave no size for.
    size: int | None


@dataclass(frozen=True)
class Resolved:
    """`repo@revision` pinned to `sha`, and what the repository holds at that commit."""

    repo: str
    revision: str
    sha: str
    files: tuple[RepoFile, ...] = ()

    @property
    def ref(self) -> SubmissionRef:
        return SubmissionRef.resolved(self.repo, self.sha)

    @property
    def declared_bytes(self) -> int:
        """The size the Hub declares for the files it sized. A lower bound on the checkout."""
        return sum(f.size for f in self.files if f.size is not None)


def resolve(repo: str, revision: str, *, api: Any = None) -> Resolved:
    """The commit `repo@revision` names, from the Hub.

    `api` is an `HfApi` or anything with its `repo_info`. A repository or revision the Hub does not
    have is a `SubmissionRejected`; a Hub that cannot be asked is a `SubmissionError`.
    """
    check_ref(repo, revision)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    with _asking_the_hub(repo, revision):
        info = api.repo_info(repo, revision=revision, repo_type=REPO_TYPE, files_metadata=True)

    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not is_commit_sha(sha):
        raise SubmissionError(f"the Hub answered {repo}@{revision} with no commit sha ({sha!r})")
    if is_commit_sha(revision) and sha != revision:
        raise SubmissionError(f"the Hub resolved commit {revision} of {repo} to another, {sha}")
    return Resolved(
        repo=repo, revision=revision, sha=sha, files=_files(getattr(info, "siblings", None))
    )


def check_ref(repo: str, revision: str) -> None:
    """`repo@revision` as something that can be resolved at all: a repo id and a revision.
    Otherwise a rejection at resolve, before the Hub or anything else is asked."""
    if not is_repo(repo):
        raise SubmissionRejected("resolve", f"{repo!r} is not a Hugging Face repo id (owner/name)")
    if not revision:
        raise SubmissionRejected("resolve", "no revision given")


@contextmanager
def _asking_the_hub(repo: str, revision: str) -> Iterator[None]:
    """What the Hub answers, as what it means for a submission: a repository or a revision it does
    not have is `SubmissionRejected`, a Hub that cannot be asked is `SubmissionError`."""
    from huggingface_hub.errors import (
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        yield
    except RepositoryNotFoundError as exc:
        raise SubmissionRejected(
            "resolve", f"{repo}@{revision}: {_not_found('repository not found', exc)}"
        ) from None
    except RevisionNotFoundError as exc:
        raise SubmissionRejected(
            "resolve", f"{repo}@{revision}: {_not_found('revision not found', exc)}"
        ) from None
    except HfHubHTTPError as exc:
        raise SubmissionError(
            f"the Hub could not resolve {repo}@{revision}: {_first_line(exc)}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - the transport's own errors: unreachable, timed out
        raise SubmissionError(f"the Hub is unreachable resolving {repo}@{revision}: {exc}") from exc


def _files(siblings: Iterable[Any] | None) -> tuple[RepoFile, ...]:
    files = []
    for entry in siblings or ():
        path = getattr(entry, "rfilename", None)
        if not isinstance(path, str):
            continue
        size = getattr(entry, "size", None)
        files.append(RepoFile(path=path, size=int(size) if isinstance(size, int) else None))
    return tuple(files)


def _not_found(what: str, exc: BaseException) -> str:
    """`what`, and the Hub's own words for it when it gave any: its 404 reads "404 Client Error.
    (Request ID: ...)" first and says what was not found lines later, in `server_message` - for a
    revision, "Invalid rev id: <name>"."""
    message = getattr(exc, "server_message", None)
    if isinstance(message, str) and message.strip() and message.strip().lower() != what:
        return f"{what}: {message.strip()}"
    return what


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
