"""Identifiers derived from the contract. Deterministic, reproducible by third parties.

A submission is a Hugging Face repository at a revision; its key, a duel's id, an event's id, a
unit's id and a unit's seed are all pure functions of published values, so anyone holding the
record can recompute them. The formulas are also written out in `spec.json` (`duel.derivation`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .canon import sha256_hex

#: Matched with `fullmatch`: Python's `$` also matches before a final newline, and a repo id or a
#: revision with a newline in it would be hashed into a key and published.
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SubmissionRef:
    """A submission as the store records it: `{key, repo, revision}` (`ModelRef` in the schema,
    whose name predates submissions being code)."""

    key: str
    repo: str
    revision: str

    @classmethod
    def make(cls, repo: str, revision: str) -> SubmissionRef:
        return cls(key=submission_key(repo, revision), repo=repo, revision=revision)

    @classmethod
    def resolved(cls, repo: str, revision: str) -> SubmissionRef:
        """`make`, for a submission being accepted: the repo id and the commit are checked."""
        if not is_repo(repo):
            raise ValueError(f"{repo!r} is not a Hugging Face repo id (owner/name)")
        if not is_commit_sha(revision):
            raise ValueError(
                f"{revision!r} is not a resolved commit sha (40 lowercase hex characters)"
            )
        return cls.make(repo, revision)

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "repo": self.repo, "revision": self.revision}

    @classmethod
    def from_dict(cls, d: dict | None) -> SubmissionRef | None:
        if not d:
            return None
        return cls(key=str(d["key"]), repo=str(d["repo"]), revision=str(d["revision"]))

    @property
    def entry(self) -> str:
        return f"{self.repo}@{self.revision}"


def submission_key(repo: str, revision: str) -> str:
    return sha256_hex(f"{repo}@{revision}")[:16]


def duel_id(
    spec_version: int, track: str, challenger: SubmissionRef, king: SubmissionRef | None
) -> str:
    king_key = king.key if king else "none"
    king_rev = king.revision if king else "none"
    return sha256_hex(
        "|".join(
            [str(spec_version), track, challenger.key, challenger.revision, king_key, king_rev]
        )
    )


def event_id(kind: str, track: str, block: int, subject: str) -> str:
    """Unique per published event: the same pair may duel again in a later block."""
    return sha256_hex("|".join(["event", kind, track, str(block), subject]))


def unit_seed(duel: str, skill: str, index: int) -> int:
    return int(sha256_hex(f"{duel}|{skill}|{index}")[:8], 16)


def unit_id(code: str, index: int) -> str:
    """`<skill code>-<index>`: the code is the two-letter `skills.<skill>.code`."""
    return f"{code}-{index:03d}"


def is_repo(value: str) -> bool:
    return bool(REPO_RE.fullmatch(value)) and len(value) <= 200


def is_commit_sha(value: str) -> bool:
    """A resolved git commit, which is what a submission is queued and published at.

    A branch or a tag is code that can change after it was queued or crowned, while the key, the
    duel id and the published record - all hashes of this string - stay as they were. An
    abbreviation is refused too: it is a different key for the same code.
    """
    return bool(COMMIT_SHA_RE.fullmatch(value))
