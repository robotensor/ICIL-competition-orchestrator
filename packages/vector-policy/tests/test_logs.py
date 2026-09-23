"""`logs.tail`: the end of a log a policy can write to, and so can replace."""

import contextlib
import os
import threading

import pytest

from vector_policy import logs


def tail_within(path, seconds=5.0):
    """`logs.tail(path)`, failing the test instead of hanging it if the call blocks."""
    result = {}
    thread = threading.Thread(target=lambda: result.update(text=logs.tail(path)), daemon=True)
    thread.start()
    thread.join(seconds)
    return thread.is_alive(), result.get("text")


def test_the_last_lines_of_the_last_bytes_are_returned(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("".join(f"line {i}\n" for i in range(1000)))
    assert logs.tail(log, lines=3) == "line 997\nline 998\nline 999"
    assert logs.tail(log, lines=1000, max_bytes=18) == "line 998\nline 999"
    assert logs.tail(tmp_path / "missing.log") == ""


def test_a_log_replaced_by_a_symlink_is_not_followed(tmp_path):
    secret = tmp_path / "benchmark-side.txt"
    secret.write_text("not the policy's to read\n")
    log = tmp_path / "serve.log"
    os.symlink(secret, log)
    assert logs.tail(log) == ""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes here")
def test_a_log_replaced_by_a_named_pipe_does_not_block(tmp_path):
    log = tmp_path / "serve.log"
    os.mkfifo(log)
    try:
        blocked, text = tail_within(log)
        assert not blocked, "reading a named pipe nobody writes to blocked"
        assert text == ""
    finally:
        with contextlib.suppress(OSError):  # frees a reader stuck in open, should there be one
            os.close(os.open(log, os.O_WRONLY | os.O_NONBLOCK))


def test_a_log_replaced_by_a_directory_is_empty(tmp_path):
    log = tmp_path / "serve.log"
    log.mkdir()
    assert logs.tail(log) == ""
