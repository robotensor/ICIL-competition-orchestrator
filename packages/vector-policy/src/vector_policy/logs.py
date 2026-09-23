"""The end of a server's log, which travels with every failure so it can be diagnosed where it lands."""

from __future__ import annotations

import os
import stat

#: How much of a log a failure carries.
TAIL_LINES = 40
TAIL_BYTES = 8192

#: The policy can write to its log's directory, so it can replace the log with a symlink to a file
#: on the benchmark's side or with a named pipe that blocks whoever opens it. Neither is followed.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


def tail(path: str | os.PathLike[str], lines: int = TAIL_LINES, max_bytes: int = TAIL_BYTES) -> str:
    """The last `lines` lines of the file at `path`, from at most its last `max_bytes`.

    "" when the file cannot be read, or is not a regular file (a symlink, a named pipe, a
    directory): a missing log must not hide the failure it would explain, and a log replaced by
    the policy must not block or leak anything.
    """
    try:
        fd = os.open(path, _OPEN_FLAGS)
    except OSError:
        return ""
    try:
        with open(fd, "rb", closefd=False) as handle:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return ""
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - max_bytes))
            data = handle.read(max_bytes)
    except OSError:
        return ""
    finally:
        os.close(fd)
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])
