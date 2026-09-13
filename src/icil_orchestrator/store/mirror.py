"""Mirror the local store to a Hugging Face dataset repo, one commit per publish.

The dashboard reads a published store from that repo (`ICIL_STORE=owner/name`), so the mirror is
what makes a result public. `huggingface_hub` is imported only when a mirror is made.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: The repository's own files rather than the store's: the Hub writes `.gitattributes`, and the
#: dataset card is written by hand. A rebuilt store must not take either of them with it.
REPO_OWNED = frozenset({".gitattributes", "README.md"})


#: Exactly the store's layout (`store/writer.py`). Nothing else is uploaded: the mirror makes a
#: dataset repository public, and a root that is not quite a store - an operator's working
#: directory, which is where `store init` keeps the signing key by default - must not publish
#: whatever else is in it. Dotfiles (the writer's locks) and half-written `.tmp` files are the
#: store's own bookkeeping and are not part of what a reader verifies.
STORE_LAYOUT = re.compile(
    r"manifest\.json"
    r"|tracks/[^/]+/(?:head|queue)\.json"
    r"|tracks/[^/]+/index-\d{4,}\.jsonl"
    r"|events/[^/]+/[0-9a-f]{8,64}\.json"
    r"|media/[0-9a-f]{1,4}/[0-9a-f]{64}\.[A-Za-z0-9]{1,8}"
)


def store_files(root: Path) -> list[str]:
    """Every file of a store that belongs in the mirror, relative to its root."""
    root = Path(root)
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and STORE_LAYOUT.fullmatch(str(p.relative_to(root)))
    )


def check_is_store(root: Path) -> list[str]:
    """The store's files, or `ValueError` if `root` is not a store. Anything else it holds is
    logged and left behind."""
    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise ValueError(f"{root} holds no manifest.json; it is not a store")
    files = store_files(root)
    known = set(files)
    beside = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and str(p.relative_to(root)) not in known
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    if beside:
        log.warning("not mirroring %d file(s) that are not the store's: %s", len(beside), beside)
    return files


class Mirror:
    def __init__(self, root: Path, repo: str, *, token: str | None = None, private: bool = False):
        from huggingface_hub import HfApi

        self.root = Path(root)
        self.repo = repo
        self.api = HfApi(token=token)
        self.api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)

    def push(self, files: list[str], message: str = "publish") -> str | None:
        """Add exactly `files` (those that exist) in one commit: what one publish touched."""
        from huggingface_hub import CommitOperationAdd

        ops = [
            CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(self.root / f))
            for f in sorted(set(files))
            if (self.root / f).exists()
        ]
        if not ops:
            return None
        info = self.api.create_commit(
            repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message
        )
        log.info("mirrored %d files to %s", len(ops), self.repo)
        return str(getattr(info, "oid", ""))

    def push_all(self, message: str = "full mirror") -> str | None:
        """Every file of the store's layout, in one commit. Only ever adds."""
        return self.push(check_is_store(self.root), message)

    def replace_all(self, message: str = "replace the store") -> str:
        """One commit that makes the repo exactly the local store: every file added, every path the
        store no longer has deleted.

        `push_all` only ever adds, which is right while a store grows and wrong when it is rebuilt:
        the previous layout's records and clips would linger beside the new ones under names nothing
        references. Deleting and adding in one commit means a reader never sees the two mixed, and
        never sees an empty store.
        """
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        local = check_is_store(self.root)
        if not local:
            raise ValueError(f"{self.root} holds no files; refusing to empty {self.repo}")
        info = self.api.repo_info(self.repo, repo_type="dataset", files_metadata=False)
        remote = {s.rfilename for s in (info.siblings or [])}
        stale = sorted(remote - set(local) - REPO_OWNED)
        ops: list[Any] = [CommitOperationDelete(path_in_repo=f) for f in stale]
        ops += [
            CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(self.root / f)) for f in local
        ]
        commit = self.api.create_commit(
            repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message
        )
        log.info("replaced %s: %d files, %d stale paths removed", self.repo, len(local), len(stale))
        return str(getattr(commit, "oid", ""))


def mirror_store(
    root: str | Path,
    repo: str,
    *,
    message: str = "publish",
    prune: bool = False,
    files: list[str] | None = None,
    token: str | None = None,
) -> int:
    """Mirror `root` to `repo`; return how many files the commit covered.

    `files` (what one publish touched) is one commit of exactly those; without it the whole store
    is uploaded, and `prune` makes the repository exactly the store.
    """
    root = Path(root)
    check_is_store(root)
    m = Mirror(root, repo, token=token)
    if prune:
        m.replace_all(message)
        return len(store_files(root))
    if files is None:
        m.push_all(message)
        return len(store_files(root))
    m.push(files, message)
    return len(files)
