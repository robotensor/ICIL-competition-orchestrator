"""`python -m vector_policy.serve`: one competitor's policy, served to one client.

    python -m vector_policy.serve --manifest PATH --address ADDR --authkey-env NAME [--log-file PATH]
                                [--idle-timeout-s SECONDS]

`ADDR` is a Unix socket path or `host:port`. The environment variable `NAME` holds the
authentication key as hex, at least 16 bytes of it; it is read once and removed from the
environment before any competitor code runs, so nothing the policy starts inherits it. With
`--log-file`, the server's log and everything the policy prints (standard output and error, native
libraries included) are appended to that file, and its tail travels with every error reply.

**Lifecycle.** The manifest is checked, the server listens, and it accepts exactly one
authenticated client. The policy is built on the first `hello` - imported with the manifest's
directory, the repository root, first on `sys.path` and as the working directory, and constructed
with the manifest's `kwargs` - and the reply carries `protocol`, `action_type` and `policy`. From
then on each `reset`, `prompt` and `act` calls the policy once and answers `ok` or `action`.

**Failure.** An exception from the policy is logged and answered with an `error` reply (`type`,
`message`, `log_tail`), and the server keeps serving: the client decides what it means. A message
that is malformed - not a JSON header, a refused dtype, a pickle - is answered with an error and
ends the session, because nothing after it can be trusted to line up. No server outlives its
client: when the client hangs up, even in the middle of a policy call that never returns, the
process exits, and a client that holds the connection with nothing to say for `--idle-timeout-s`
(30 minutes by default; a policy call in progress is not idle) is taken to have gone. (A call stuck in native code that holds the GIL cannot be interrupted from Python;
the container around the server is the last resort.)

**Exit status.** The process exits as soon as the session ends, however it ends, without waiting
for threads the policy started. 0 after `close`, when the client hangs up or when it was idle too
long; 1 when the policy
could not be built, a malformed message ended the session or anything else went wrong (such as a
`KeyboardInterrupt`, which is not answered); 2 when serving never started (arguments, key,
manifest or address).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import select
import socket
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np

from . import logs, wire
from .errors import ManifestError, WireError
from .manifest import Manifest
from .manifest import load as load_manifest
from .policy import ACTION_TYPES

log = logging.getLogger("vector_policy.serve")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

#: The longest exception message an error reply carries.
MESSAGE_CHARS = 4000
#: How often the hang-up watch looks at the connection while a policy call runs.
WATCH_SLICE_S = 0.1
#: How long a client may hold the connection with nothing to say before it is taken to have gone.
#: Generous: a benchmark building a scene or writing a video between calls is working, not idle.
IDLE_TIMEOUT_S = 1800.0


class _HungUp(Exception):
    """The client went away while the server was answering."""


class _SessionOver(Exception):
    """The session cannot go on; `status` is the exit status."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def build_policy(manifest: Manifest) -> Any:
    """The manifest's policy, imported from its repository root and constructed with its kwargs.

    Changes the working directory to the repository root and puts it first on `sys.path`.
    """
    root = str(manifest.root)
    os.chdir(root)
    with contextlib.suppress(ValueError):
        sys.path.remove(root)
    sys.path.insert(0, root)
    importlib.invalidate_caches()
    module = importlib.import_module(manifest.module)
    cls = getattr(module, manifest.attribute, None)
    if cls is None:
        raise AttributeError(f"module {manifest.module!r} has no attribute {manifest.attribute!r}")
    policy = cls(**manifest.kwargs)
    problems = []
    action_type = getattr(policy, "action_type", None)
    if action_type not in ACTION_TYPES:
        problems.append(f"action_type is {action_type!r}, not one of {', '.join(ACTION_TYPES)}")
    for method in ("reset", "set_demonstration", "act"):
        if not callable(getattr(policy, method, None)):
            problems.append(f"it has no {method}() method")
    if problems:
        raise TypeError(f"{manifest.policy} is not a Policy: {'; '.join(problems)}")
    return policy


def checked_action(result: Any) -> dict[str, Any]:
    """What `act` returned, if it is a mapping of names to arrays with an `action` in it."""
    if not isinstance(result, Mapping):
        raise TypeError(f"act() returned {type(result).__name__}, not a mapping of named arrays")
    if "action" not in result:
        raise ValueError(f"act() returned no 'action', only {sorted(map(str, result))}")
    shape = np.shape(result["action"])
    if len(shape) not in (1, 2) or 0 in shape:
        raise ValueError(f"act() returned an 'action' of shape {shape}, not (A,) or (H, A)")
    return dict(result)


class _HangupWatch:
    """Exits the process if the client hangs up while a policy call is running.

    The session thread is inside competitor code during a call, so it cannot notice the client
    leave. This thread peeks at the connection, through a duplicate of its descriptor, without
    consuming anything: an end-of-file while a call is running means nobody is waiting for the
    answer, and the server exits rather than outlive its client.
    """

    def __init__(self, conn: Any) -> None:
        fd = conn.fileno()
        self._sock = socket.socket(fileno=os.dup(fd))
        # A socket object made while a default timeout is set turns the shared descriptor
        # non-blocking, which would break the connection's own reads.
        os.set_blocking(fd, True)
        self._busy = threading.Event()
        self._op = ""
        threading.Thread(target=self._run, name="vector-policy-hangup-watch", daemon=True).start()

    @contextlib.contextmanager
    def __call__(self, op: str) -> Iterator[None]:
        self._op = op
        self._busy.set()
        try:
            yield
        finally:
            self._busy.clear()

    def _run(self) -> None:
        while True:
            self._busy.wait()
            while self._busy.is_set():
                try:
                    ready, _, _ = select.select([self._sock], [], [], WATCH_SLICE_S)
                except (OSError, ValueError):
                    return
                if not ready:
                    continue
                try:
                    data = self._sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    data = b""
                if data:
                    time.sleep(WATCH_SLICE_S)  # a message waits; it is read after this call
                elif self._busy.is_set():
                    log.warning(
                        "the client hung up during %s; exiting without its answer", self._op
                    )
                    _flush()
                    os._exit(EXIT_OK)


class Session:
    """One client, from `hello` to `close` or hang-up."""

    def __init__(
        self,
        conn: Any,
        manifest: Manifest,
        log_file: Path | None,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
    ) -> None:
        self.conn = conn
        self.idle_timeout_s = idle_timeout_s
        self.manifest = manifest
        self.log_file = log_file
        self.policy: Any = None
        self.action_type: str | None = None
        self.watch = _HangupWatch(conn)

    def run(self) -> int:
        try:
            while True:
                try:
                    if not self.conn.poll(self.idle_timeout_s):
                        log.warning(
                            "the client said nothing for %gs; ending the session",
                            self.idle_timeout_s,
                        )
                        return EXIT_OK
                    op, fields, arrays = wire.recv(self.conn)
                except (EOFError, OSError):
                    log.info("the client hung up")
                    return EXIT_OK
                except Exception as exc:  # WireError, or whatever else a hostile message raises
                    why = str(exc) if isinstance(exc, WireError) else f"{type(exc).__name__}: {exc}"
                    log.error("malformed message, ending the session: %s", why)
                    self._error("WireError", why)
                    return EXIT_FAILED
                self._dispatch(op, fields, arrays)
        except _HungUp:
            log.info("the client hung up")
            return EXIT_OK
        except _SessionOver as over:
            return over.status

    def _dispatch(self, op: str, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        handler = getattr(self, f"_op_{op}", None)
        if op not in wire.CLIENT_OPS or handler is None:
            self._error(
                "WireError", f"unknown op {op!r}; a client sends {', '.join(wire.CLIENT_OPS)}"
            )
        elif self.policy is None and op not in ("hello", "close"):
            self._error("WireError", f"{op} before hello")
        else:
            handler(fields, arrays)

    # -- operations -------------------------------------------------------------------------

    def _op_hello(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        log.info("hello from %r", fields.get("client"))
        if self.policy is None:
            ok, policy = self._call("hello", build_policy, self.manifest)
            if not ok:
                raise _SessionOver(EXIT_FAILED)
            self.policy, self.action_type = policy, policy.action_type
            log.info("serving %s, action_type %r", self.manifest.policy, self.action_type)
        self._send(
            "ok",
            {
                "protocol": wire.PROTOCOL_VERSION,
                "action_type": self.action_type,
                "policy": self.manifest.policy,
            },
        )

    def _op_reset(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        seed = fields.get("seed")
        if type(seed) is not int:
            self._error("WireError", f"reset: seed must be an integer, not {seed!r}")
        elif self._call("reset", self.policy.reset, seed)[0]:
            self._send("ok")

    def _op_prompt(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        info = fields.get("info", {})
        if not isinstance(info, dict):
            self._error("WireError", f"prompt: info must be an object, not {type(info).__name__}")
        elif self._call("prompt", self.policy.set_demonstration, arrays, info)[0]:
            self._send("ok")

    def _op_act(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        # Encoding is part of the guarded call: what act returned is the policy's to get right.
        ok, frames = self._call("act", self._act_frames, arrays)
        if ok:
            self._send_frames(frames)

    def _act_frames(self, observation: dict[str, np.ndarray]) -> list[bytes]:
        return wire.encode("action", arrays=checked_action(self.policy.act(observation)))

    def _op_close(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close) and not self._call("close", close)[0]:
            raise _SessionOver(EXIT_OK)
        with contextlib.suppress(_HungUp):
            self._send("ok")
        log.info("closed")
        raise _SessionOver(EXIT_OK)

    # -- plumbing ---------------------------------------------------------------------------

    def _call(self, op: str, fn: Any, *args: Any) -> tuple[bool, Any]:
        """`(True, fn(*args))`, or `(False, None)` once the exception has been answered."""
        try:
            with self.watch(op):
                return True, fn(*args)
        except (Exception, SystemExit) as exc:
            log.error("%s raised", op, exc_info=True)
            self._error(type(exc).__name__, f"{op}: {exc}")
            return False, None

    def _error(self, kind: str, message: str) -> None:
        if len(message) > MESSAGE_CHARS:
            message = message[:MESSAGE_CHARS] + " [...]"
        self._send("error", {"type": kind, "message": message, "log_tail": self._log_tail()})

    def _send(self, op: str, fields: Mapping[str, Any] | None = None) -> None:
        self._send_frames(wire.encode(op, fields))

    def _send_frames(self, frames: list[bytes]) -> None:
        try:
            for frame in frames:
                self.conn.send_bytes(frame)
        except (EOFError, OSError):
            raise _HungUp from None

    def _log_tail(self) -> str:
        if self.log_file is None:
            return ""
        _flush()
        return logs.tail(self.log_file)


def _flush() -> None:
    for handler in log.handlers:
        with contextlib.suppress(Exception):
            handler.flush()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()


def _redirect_output(path: Path) -> None:
    """Append this process's standard output and error, at the descriptor level, to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    _flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(line_buffering=True)


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s vector_policy.serve[%(process)d] %(levelname)s %(message)s")
    )
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False  # a policy that configures the root logger does not silence the server


def _accept(listener: Listener) -> Any:
    """The first client that authenticates. A wrong key is logged and the next client awaited."""
    while True:
        try:
            return listener.accept()
        except AuthenticationError as exc:
            log.warning("refused a client: %s", exc)
        except (EOFError, ConnectionError) as exc:
            log.warning("a client left during authentication: %s", exc)
        except OSError as exc:
            if exc.errno is not None:
                raise  # the listener itself failed
            log.warning("a client sent nonsense during authentication: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vector_policy.serve",
        description="Serve the policy named by an icil.yaml to one client, then exit.",
    )
    parser.add_argument("--manifest", required=True, help="the competitor repository's icil.yaml")
    parser.add_argument(
        "--address", required=True, help="a Unix socket path, or host:port, to listen on"
    )
    parser.add_argument(
        "--authkey-env",
        required=True,
        metavar="NAME",
        help="the environment variable holding the key as hex; removed once read",
    )
    parser.add_argument("--log-file", help="append the log and the policy's output to this file")
    parser.add_argument(
        "--idle-timeout-s",
        type=_positive_seconds,
        default=IDLE_TIMEOUT_S,
        metavar="SECONDS",
        help=f"end the session after a client says nothing this long (default {IDLE_TIMEOUT_S:g})",
    )
    return parser


def _positive_seconds(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = float("nan")
    if not seconds > 0 or seconds == float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number of seconds")
    return seconds


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_file = Path(args.log_file).absolute() if args.log_file else None
    if log_file is not None:
        try:
            _redirect_output(log_file)
        except OSError as exc:
            print(f"vector_policy.serve: cannot write the log file: {exc}", file=sys.stderr)
            return EXIT_USAGE
    _configure_logging()

    hexkey = os.environ.pop(args.authkey_env, None)
    try:
        authkey = bytes.fromhex(hexkey or "")
    except ValueError:
        authkey = b""
    if not authkey:
        log.error("environment variable %s does not hold a hex authkey", args.authkey_env)
        return EXIT_USAGE
    if len(authkey) < wire.MIN_AUTHKEY_BYTES:
        log.error(
            "the authkey must be at least %d bytes, not %d", wire.MIN_AUTHKEY_BYTES, len(authkey)
        )
        return EXIT_USAGE

    try:
        family, target = wire.parse_address(args.address)
        manifest = load_manifest(args.manifest)
    except (WireError, ManifestError) as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    if family == "AF_UNIX":
        target = str(Path(target).absolute())

    try:
        listener = Listener(target, family=family, authkey=authkey)
    except OSError as exc:
        log.error("cannot listen on %s: %s", args.address, exc)
        return EXIT_USAGE
    log.info("listening on %s to serve %s to one client", args.address, manifest.policy)
    try:
        conn = _accept(listener)
    finally:
        listener.close()
    try:
        return Session(conn, manifest, log_file, args.idle_timeout_s).run()
    finally:
        _flush()
        with contextlib.suppress(OSError):
            conn.close()


def _exit(status: int) -> None:
    """Exit now, without waiting for threads the policy may have left running."""
    _flush()
    os._exit(status)


def _serve_and_exit() -> None:
    """`main`, then `_exit` however it ended, an escaping exception included."""
    status = EXIT_FAILED
    try:
        status = main()
    except SystemExit as exc:  # argparse, for --help and for arguments it refuses
        code = EXIT_OK if exc.code is None else exc.code
        status = code if isinstance(code, int) else EXIT_FAILED
    except BaseException:
        log.exception("serving failed")
    finally:
        _exit(status)


if __name__ == "__main__":
    _serve_and_exit()
