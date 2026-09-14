"""A submission's repository at its commit, in a cache addressed by that commit.

    <cache root>/<sha>/repo/          the checkout, plain files, no symlink of the Hub's cache
    <cache root>/<sha>/fetched.json   what it is and how big; written last, so its presence means
                                      the checkout is complete

A commit sha names one tree, so the same sha fetched twice is the same bytes and is downloaded
once. The checkout is filled next to its final place and moved there in one rename, under a lock
per sha, so two processes fetching the same submission cannot half-fill each other's directory.
`spec.submission.max_repo_bytes` is enforced twice: on the sizes the Hub declares before a byte is
downloaded, and on the bytes actually on disk after. A file the Hub declares no size for is not
downloaded at all, since the first check could not see it.

`HubFetcher` is the real thing; `LocalFetcher` takes a directory in its place, for the tests and
for `submission check --local`, and addresses it by a hash of its tree, so it too is pinned to
exactly the content that was checked.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..store.records import now_iso
from ..store.writer import atomic_write_json
from .errors import SubmissionError, SubmissionRejected
from .resolve import REPO_TYPE, RepoFile, Resolved, check_ref, resolve

FETCHED_FILE = "fetched.json"
REPO_DIR = "repo"
PARTIAL_DIR = "partial"
#: What huggingface_hub leaves in a checkout it wrote: its own bookkeeping, not the repository.
HUB_CACHE_DIR = ".cache"
#: What a local directory holds that a push to the Hub would not carry.
LOCAL_IGNORED = frozenset({".git", "__pycache__"})


@dataclass(frozen=True)
class Fetched:
    """A submission's checkout in the cache."""

    resolved: Resolved
    root: Path
    bytes: int
    files: int
    #: True when the checkout was already there and nothing was downloaded.
    cached: bool


class RepoCache:
    """The cache directory and the one way into it, `store`."""

    def __init__(self, root: str | os.PathLike[str], max_bytes: int) -> None:
        self.root = Path(root)
        self.max_bytes = int(max_bytes)

    def checkout(self, sha: str) -> Path:
        return self.root / sha / REPO_DIR

    def lookup(self, resolved: Resolved) -> Fetched | None:
        """The complete checkout of `resolved.sha`, if the cache holds one."""
        marker = self.root / resolved.sha / FETCHED_FILE
        try:
            doc = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        root = self.checkout(resolved.sha)
        if not isinstance(doc, dict) or not root.is_dir():
            return None
        return Fetched(
            resolved=resolved,
            root=root,
            bytes=int(doc.get("bytes", 0)),
            files=int(doc.get("files", 0)),
            cached=True,
        )

    def store(self, resolved: Resolved, fill: Callable[[Path], None]) -> Fetched:
        """`fill` writes the checkout into a directory; it becomes `checkout(sha)` if it fits.

        Anything `fill` leaves that is not the repository is the caller's to remove before
        returning. Whatever goes wrong, nothing half-done stays behind.
        """
        entry = self.root / resolved.sha
        entry.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root / f"{resolved.sha}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            cached = self.lookup(resolved)
            if cached is not None:
                return cached
            return self._fill(resolved, entry, fill)
        finally:
            os.close(fd)

    def _fill(self, resolved: Resolved, entry: Path, fill: Callable[[Path], None]) -> Fetched:
        partial = entry / PARTIAL_DIR
        final = self.checkout(resolved.sha)
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        try:
            fill(partial)
            size, count = measure(partial)
            if size > self.max_bytes:
                raise SubmissionRejected(
                    "fetch",
                    f"{resolved.repo}@{resolved.sha} is {size} bytes on disk, over "
                    f"max_repo_bytes ({self.max_bytes})",
                )
            shutil.rmtree(final, ignore_errors=True)
            partial.rename(final)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        atomic_write_json(
            entry / FETCHED_FILE,
            {
                "repo": resolved.repo,
                "revision": resolved.revision,
                "sha": resolved.sha,
                "bytes": size,
                "files": count,
                "fetched_at": now_iso(),
            },
        )
        return Fetched(resolved=resolved, root=final, bytes=size, files=count, cached=False)


class HubFetcher:
    """Resolve on the Hub and download with `snapshot_download`, into a `RepoCache`."""

    def __init__(
        self, cache: RepoCache, *, api: Any = None, download: Callable[..., Any] | None = None
    ) -> None:
        self.cache = cache
        self.api = api
        self._download = download

    def resolve(self, repo: str, revision: str) -> Resolved:
        return resolve(repo, revision, api=self.api)

    def fetch(self, resolved: Resolved) -> Fetched:
        cached = self.cache.lookup(resolved)
        if cached is not None:
            return cached
        # The cap is held to before the download on what the Hub declares, which with
        # `files_metadata` is every file's size. A file it gave no size for cannot be bounded
        # before it is on disk, so it is not downloaded: the Hub's answer, not the entry's doing.
        unsized = [f.path for f in resolved.files if f.size is None]
        if unsized:
            shown = ", ".join(unsized[:3]) + (", ..." if len(unsized) > 3 else "")
            raise SubmissionError(
                f"the Hub declared no size for {len(unsized)} file(s) of "
                f"{resolved.repo}@{resolved.sha} ({shown}), so the download cannot be held to "
                f"max_repo_bytes before it happens"
            )
        declared = resolved.declared_bytes
        if declared > self.cache.max_bytes:
            raise SubmissionRejected(
                "fetch",
                f"{resolved.repo}@{resolved.sha} declares {declared} bytes, over "
                f"max_repo_bytes ({self.cache.max_bytes})",
            )
        return self.cache.store(resolved, lambda into: self._snapshot(resolved, into))

    def _snapshot(self, resolved: Resolved, into: Path) -> None:
        download = self._download
        if download is None:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            download = snapshot_download
        try:
            download(resolved.repo, revision=resolved.sha, repo_type=REPO_TYPE, local_dir=str(into))
        except SubmissionRejected:
            raise
        except Exception as exc:  # noqa: BLE001 - whatever the transport raised: not the entry's
            raise SubmissionError(
                f"downloading {resolved.repo}@{resolved.sha} failed: {type(exc).__name__}: {exc}"
            ) from exc
        shutil.rmtree(into / HUB_CACHE_DIR, ignore_errors=True)


class LocalFetcher:
    """A directory in place of the Hub: resolved to the hash of its tree, copied into the cache.

    The revision asked for is recorded as given; the sha is the tree's, so the record still names
    exactly the content that ran.
    """

    def __init__(self, cache: RepoCache, directory: str | os.PathLike[str]) -> None:
        self.cache = cache
        self.directory = Path(directory)

    def resolve(self, repo: str, revision: str) -> Resolved:
        check_ref(repo, revision)  # the ref is recorded as given, so it is held to the same shape
        if not self.directory.is_dir():
            raise SubmissionError(f"{self.directory} is not a directory")
        files = tuple(
            RepoFile(path=rel.as_posix(), size=os.lstat(self.directory / rel).st_size)
            for rel in _tree_files(self.directory)
        )
        return Resolved(repo=repo, revision=revision, sha=tree_hash(self.directory), files=files)

    def fetch(self, resolved: Resolved) -> Fetched:
        cached = self.cache.lookup(resolved)
        if cached is not None:
            return cached

        def copy(into: Path) -> None:
            shutil.copytree(
                self.directory,
                into,
                symlinks=True,
                ignore=lambda _, names: [n for n in names if n in LOCAL_IGNORED],
                dirs_exist_ok=True,
            )

        return self.cache.store(resolved, copy)


def measure(root: Path) -> tuple[int, int]:
    """Bytes and files under `root`, without following symlinks: a link counts for its own size,
    not for what it points at, which may be outside the checkout."""
    size = count = 0
    for path in _walk(root):
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            size += info.st_size
            count += 1
    return size, count


def tree_hash(directory: Path) -> str:
    """A 40-hex address of a directory's content: the files, their modes and their bytes.

    Like a git tree hash, it changes when any file changes and is the same for two copies of the
    same tree, so a local directory is pinned the way a commit pins a repository.
    """
    digest = hashlib.sha1()
    for rel in _tree_files(directory):
        path = directory / rel
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            content = hashlib.sha1(os.readlink(path).encode("utf-8", "surrogateescape"))
            mode = "120000"
        else:
            content = hashlib.sha1()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    content.update(chunk)
            mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
        digest.update(f"{mode} {content.hexdigest()} {rel.as_posix()}\n".encode())
    return digest.hexdigest()


def _tree_files(directory: Path) -> list[Path]:
    """Every regular file and symlink under `directory`, relative, sorted, minus LOCAL_IGNORED."""
    files = []
    for path in _walk(directory, ignored=LOCAL_IGNORED):
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            files.append(path.relative_to(directory))
    return sorted(files, key=lambda p: p.as_posix())


def _walk(root: Path, ignored: frozenset[str] = frozenset()) -> list[Path]:
    """Every entry under `root` that is not a directory, symlinks to directories included: a link
    is an entry of the tree, and nothing here follows one."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        keep = []
        for name in sorted(dirnames):
            if name in ignored:
                continue
            if os.path.islink(here / name):
                found.append(here / name)
            else:
                keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            found.append(here / name)
    return found
