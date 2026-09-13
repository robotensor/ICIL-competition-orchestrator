"""The end of a server's log, which travels with every failure so it can be diagnosed where it lands."""

from __future__ import annotations

import os

#: How much of a log a failure carries.
TAIL_LINES = 40
TAIL_BYTES = 8192


def tail(path: str | os.PathLike[str], lines: int = TAIL_LINES, max_bytes: int = TAIL_BYTES) -> str:
    """The last `lines` lines of the file at `path`, from at most its last `max_bytes`.

    "" when the file cannot be read: a missing log must not hide the failure it would explain.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - max_bytes))
            data = handle.read()
    except OSError:
        return ""
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])
