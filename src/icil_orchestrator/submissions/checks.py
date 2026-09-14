"""What a checkout is looked at for before any of it is built or run: its manifest.

`icil_policy.manifest.load` reads the file and checks its schema. Before it is called, the file
itself is checked to be a plain entry of the repository: a git checkout can hold a symbolic link,
and a manifest that is one - or a named pipe, or a directory - would have the orchestrator read
something other than the repository's bytes, or block on them. The requirements file the manifest
names gets the same treatment. Nothing here imports what the manifest names.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from icil_policy.errors import ManifestError
from icil_policy.manifest import Manifest
from icil_policy.manifest import load as load_manifest

from .errors import SubmissionRejected

STEP = "manifest"


def check_repository(root: Path, spec: Any) -> Manifest:
    """The checkout's manifest, valid for this competition; `SubmissionRejected` otherwise."""
    submission = spec.submission
    name = str(submission["manifest"])
    path = regular_file_inside(root, name, what=f"the manifest {name}")
    try:
        manifest = load_manifest(path)
    except ManifestError as exc:
        problems = "; ".join(exc.problems)
        raise SubmissionRejected(STEP, f"{name}: {problems}") from None
    wanted = int(submission["manifest_api"])
    if manifest.api != wanted:
        raise SubmissionRejected(
            STEP, f"{name}: api {manifest.api} is not this competition's {wanted}"
        )
    if manifest.requirements is not None:
        regular_file_inside(root, manifest.requirements, what="requirements")
    return manifest


def regular_file_inside(root: Path, relative: str, *, what: str) -> Path:
    """`root/relative`, when every component is a plain directory entry and the last a regular
    file. A symbolic link anywhere on the way is refused rather than followed."""
    pure = PurePosixPath(relative)
    parts = pure.parts
    if pure.is_absolute() or not parts or any(p in ("..", ".") for p in parts):
        raise SubmissionRejected(STEP, f"{what}: {relative!r} is not a path inside the repository")
    current = root
    info = None
    for part in parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            raise SubmissionRejected(
                STEP, f"{what}: {relative!r} is not in the repository"
            ) from None
        except OSError as exc:
            raise SubmissionRejected(STEP, f"{what}: {relative!r} cannot be read: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise SubmissionRejected(
                STEP, f"{what}: {relative!r} goes through a symbolic link ({part}); refused"
            )
    assert info is not None
    if not stat.S_ISREG(info.st_mode):
        raise SubmissionRejected(STEP, f"{what}: {relative!r} is not a regular file")
    return current
