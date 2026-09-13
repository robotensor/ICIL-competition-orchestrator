"""`RemotePolicy`: what a benchmark drives a served policy with.

    with RemotePolicy(address, authkey, timeout_s=60.0, log_file=server_log) as policy:
        policy.hello()                              # {"protocol": 1, "action_type": ..., "policy": ...}
        policy.set_demonstration(prompt_arrays, info)
        policy.reset(seed)
        action = policy.act(observation)["action"]  # (A,) or a chunk (H, A)

It needs numpy and the standard library only. A benchmark imports it; the competitor's side runs
`python -m icil_policy.serve`.

**Every way the policy can fail raises `PolicyUnavailable`**: an error reply, a call that does not
finish within `timeout_s`, a server that hangs up or was never there, a key it refuses, a reply
that is malformed. When `log_file` names the server's log, the exception's message ends with its
tail; otherwise with the tail the server sent in its error reply, if any. A benchmark catches that
one exception and decides what it costs the unit.

**Timeouts are hard.** Connecting (retried while nothing listens yet) and authenticating share
one `timeout_s`, and each call gets its own, covering both sending the request and receiving the
whole reply. When one runs out the socket is shut down, which unblocks whatever was waiting and
tells the server its client has gone; the server exits.

After an error reply the connection stays usable - the server keeps serving. After any other
failure it is closed, and every later call raises `PolicyUnavailable` at once. A value that cannot
be sent at all, such as an object array, is the caller's mistake: `WireError`, before anything is
sent, and the connection is untouched. One `RemotePolicy` serves one thread at a time.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from collections.abc import Iterator, Mapping
from multiprocessing import AuthenticationError
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
from typing import Any

import numpy as np

from . import __version__, logs, wire
from .errors import PolicyUnavailable, WireError
from .policy import ACTION_TYPES

__all__ = ["PolicyUnavailable", "RemotePolicy"]

#: The most array bytes a reply may carry. An action is small; this bounds a hostile server.
MAX_REPLY_BYTES = 1 << 30
#: How long to wait between attempts to reach an address nothing listens on yet.
RETRY_S = 0.05


class RemotePolicy:
    """A policy served by `python -m icil_policy.serve` at `address`, reached with `authkey`."""

    def __init__(
        self,
        address: str,
        authkey: bytes,
        *,
        timeout_s: float = 60.0,
        log_file: str | os.PathLike[str] | None = None,
    ) -> None:
        if not isinstance(authkey, (bytes, bytearray)) or not authkey:
            raise TypeError("authkey must be non-empty bytes")
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be positive, not {timeout_s!r}")
        self.address = address
        self.timeout_s = float(timeout_s)
        self.log_file = log_file
        #: The served policy's action type, known once `hello` has been answered.
        self.action_type: str | None = None
        self._sock: socket.socket | None = None
        self._conn: Connection | None = None
        self._closed = False
        self._lock = threading.Lock()
        try:
            family, target = wire.parse_address(address)
        except WireError as exc:
            raise self._unavailable("connect", str(exc)) from None
        self._connect(family, target, bytes(authkey))

    # -- the protocol -----------------------------------------------------------------------

    def hello(self) -> dict[str, Any]:
        """Greet the server, which builds the policy now; its `protocol`, `action_type`, `policy`."""
        fields, _ = self._call("hello", {"client": f"icil-policy {__version__}"})
        protocol, action_type = fields.get("protocol"), fields.get("action_type")
        if type(protocol) is not int or protocol != wire.PROTOCOL_VERSION:
            self._abandon()
            raise self._unavailable(
                "hello", f"the server speaks protocol {protocol!r}, not {wire.PROTOCOL_VERSION}"
            )
        if action_type not in ACTION_TYPES:
            self._abandon()
            raise self._unavailable("hello", f"the policy declares action_type {action_type!r}")
        self.action_type = action_type
        return dict(fields)

    def reset(self, seed: int) -> None:
        """Start an episode."""
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, not {seed!r}")
        self._call("reset", {"seed": int(seed)})

    def set_demonstration(self, arrays: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        """Hand over the demonstration: named arrays and public `info` fields, never `meta`."""
        if not isinstance(info, Mapping):
            raise TypeError(f"info must be a mapping, not {type(info).__name__}")
        self._call("prompt", {"info": dict(info)}, arrays)

    def act(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The policy's answer to one observation: at least `action`, of shape (A,) or (H, A)."""
        _, arrays = self._call("act", {}, observation, expect="action")
        action = arrays.get("action")
        if action is None or action.ndim not in (1, 2):
            shape = None if action is None else action.shape
            raise self._unavailable("act", f"the reply holds no usable 'action' (shape {shape})")
        return arrays

    def close(self) -> None:
        """Say `close`, then close the connection whatever the answer. Idempotent."""
        if self._closed:
            return
        try:
            if self._conn is not None:
                self._call("close")
        finally:
            self._closed = True
            self._abandon()

    def __enter__(self) -> RemotePolicy:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            with contextlib.suppress(PolicyUnavailable):
                self.close()

    # -- the connection ---------------------------------------------------------------------

    def _connect(self, family: str, target: Any, authkey: bytes) -> None:
        deadline = time.monotonic() + self.timeout_s
        while True:
            sock = socket.socket(getattr(socket, family), socket.SOCK_STREAM)
            try:
                sock.settimeout(max(deadline - time.monotonic(), 0.001))
                sock.connect(target)
                break
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                sock.close()
                if time.monotonic() >= deadline:
                    raise self._unavailable(
                        "connect",
                        f"nothing listened at {self.address} within {self.timeout_s:g}s ({exc})",
                    ) from None
                time.sleep(min(RETRY_S, max(deadline - time.monotonic(), 0)))
            except OSError as exc:
                sock.close()
                raise self._unavailable("connect", f"cannot reach {self.address}: {exc}") from None
        sock.settimeout(None)
        self._sock = sock
        self._conn = Connection(os.dup(sock.fileno()))
        failure = None
        with self._deadline(deadline - time.monotonic()) as expired:
            try:
                answer_challenge(self._conn, authkey)
                deliver_challenge(self._conn, authkey)
            except AuthenticationError as exc:
                failure = f"the server at {self.address} refused this key: {exc}"
            except (EOFError, OSError) as exc:
                failure = f"the server hung up while authenticating: {_describe(exc)}"
        if expired.is_set():
            failure = f"authentication did not finish within {self.timeout_s:g}s"
        if failure is not None:
            self._abandon()
            raise self._unavailable("connect", failure)

    def _call(
        self,
        op: str,
        fields: Mapping[str, Any] | None = None,
        arrays: Mapping[str, Any] | None = None,
        *,
        expect: str = "ok",
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        with self._lock:
            conn = self._conn
            if conn is None:
                why = "closed" if self._closed else "closed after an earlier failure"
                raise self._unavailable(op, f"the connection to {self.address} is {why}")
            frames = wire.encode(op, fields, arrays)  # refused here, before anything is sent
            failure = None
            with self._deadline(self.timeout_s) as expired:
                try:
                    for frame in frames:
                        conn.send_bytes(frame)
                    reply_op, reply_fields, reply_arrays = wire.recv(
                        conn, max_bytes=MAX_REPLY_BYTES
                    )
                except (EOFError, OSError) as exc:
                    failure = f"the policy went away: {_describe(exc)}"
                except WireError as exc:
                    failure = f"malformed reply: {exc}"
                except Exception as exc:  # whatever else a hostile reply makes reading raise
                    failure = f"malformed reply: {_describe(exc)}"
            if expired.is_set():
                failure = f"no answer within {self.timeout_s:g}s"
            if failure is not None:
                self._abandon()
                raise self._unavailable(op, failure)

        if reply_op == "error":
            kind = str(reply_fields.get("type") or "Error")
            message = str(reply_fields.get("message") or "")
            tail = reply_fields.get("log_tail")
            raise self._unavailable(
                op,
                f"{kind}: {message}",
                remote_type=kind,
                server_tail=tail if isinstance(tail, str) else "",
            )
        if reply_op != expect:
            self._abandon()
            raise self._unavailable(op, f"the server answered {reply_op!r}, not {expect!r}")
        return reply_fields, reply_arrays

    @contextlib.contextmanager
    def _deadline(self, seconds: float) -> Iterator[threading.Event]:
        """Shut the socket down if the block has not finished within `seconds`.

        A shutdown, unlike a close, wakes a thread blocked reading or writing the socket. The
        event is set if it fired: the block's result, if any, is then not to be trusted.
        """
        expired = threading.Event()
        sock = self._sock

        def expire() -> None:
            expired.set()
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)

        timer = threading.Timer(max(seconds, 0.0), expire)
        timer.daemon = True
        timer.start()
        try:
            yield expired
        finally:
            timer.cancel()

    def _abandon(self) -> None:
        """Close the connection. The server sees its client hang up and exits."""
        conn, sock = self._conn, self._sock
        self._conn = self._sock = None
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                sock.close()

    def _unavailable(
        self,
        op: str,
        message: str,
        *,
        remote_type: str | None = None,
        server_tail: str = "",
    ) -> PolicyUnavailable:
        tail = logs.tail(self.log_file) if self.log_file is not None else ""
        return PolicyUnavailable(
            f"{op}: {message}", op=op, remote_type=remote_type, log_tail=tail or server_tail
        )


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
