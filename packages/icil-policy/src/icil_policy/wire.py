"""The message format between a benchmark and a served policy: named arrays and JSON fields.

Ported from `icilval.model.wire` (ICIL-LiberoGen-bench, branch `milestone-two-fields`).

**Why its own format.** A benchmark's own transport carries that benchmark's types, and a policy
that spoke it would be written for one benchmark. This format carries named arrays and JSON fields
and nothing else: the benchmark and the policy agree on the names (`frames_head`, `qpos`,
`actions`, ...) between themselves, and nothing here reads them.

**What it deliberately cannot do.** It never pickles. `Connection.send` and `Connection.recv`
pickle, and unpickling runs code chosen by whoever wrote the bytes - on one side of this socket is
a competitor's code - so only `send_bytes` and `recv_bytes` are used. Arrays are bool, integer or
float, little-endian; an object array, or any other dtype, is refused before it is sent and when a
header announces it.

**A message** is one JSON header frame,

    {"protocol": 1, "op": "act", "fields": {...}, "arrays": [{"name", "dtype", "shape"}, ...]}

followed by one raw little-endian frame per array, in the order the header lists them. The header
is validated in full before any array frame is read, and each frame is read with the size its
description implies as the limit, so a hostile header cannot size an allocation it does not pay
for.

**Ops.** A client sends `hello` (fields: `client`), `reset` (fields: `seed`), `prompt` (arrays:
the demonstration; fields: `info`), `act` (arrays: the observation) and `close`. The server answers
each with `ok` (the reply to `hello` carries `protocol`, `action_type` and `policy`), `action`
(arrays, `action` among them) or `error` (fields: `type`, `message`, `log_tail`).

**Addresses** are a Unix socket path or `host:port`, both authenticated with a shared key by
`multiprocessing.connection`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .errors import WireError

__all__ = [
    "CLIENT_OPS",
    "DTYPES",
    "MAX_HEADER_BYTES",
    "MAX_MESSAGE_BYTES",
    "PROTOCOL_VERSION",
    "REPLY_OPS",
    "WireError",
    "encode",
    "parse_address",
    "recv",
    "send",
]

#: Bumped whenever a message changes shape. Every header carries it and every receiver checks it;
#: the reply to `hello` repeats it so a client can refuse a server before the first real call.
PROTOCOL_VERSION = 1

#: Every dtype an array may have on the wire, as numpy spells it little-endian. Anything else -
#: object above all, but also strings, datetimes, structured and big-endian dtypes - is refused.
DTYPES = frozenset(
    np.dtype(name).newbyteorder("<").str
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
)

#: What a client may send.
CLIENT_OPS = ("hello", "reset", "prompt", "act", "close")
#: What a server may answer.
REPLY_OPS = ("ok", "action", "error")

#: The longest header frame accepted. Headers hold names, shapes and a few public fields.
MAX_HEADER_BYTES = 4 << 20
#: The most array bytes one message may carry, unless the receiver asks for less.
MAX_MESSAGE_BYTES = 16 << 30

_HEADER_KEYS = frozenset({"protocol", "op", "fields", "arrays"})
_ARRAY_KEYS = frozenset({"name", "dtype", "shape"})


def parse_address(address: str) -> tuple[str, Any]:
    """`(family, target)` for `multiprocessing.connection`: `host:port`, or a Unix socket path.

    `host:port` is an address with no `/` whose part after the last `:` is a port number;
    anything else is a path.
    """
    if not isinstance(address, str) or not address:
        raise WireError(f"address {address!r} is neither a Unix socket path nor host:port")
    host, sep, port = address.rpartition(":")
    if sep and host and port.isdigit() and "/" not in address:
        if not 0 < int(port) < 65536:
            raise WireError(f"address {address!r}: port out of range")
        return "AF_INET", (host, int(port))
    return "AF_UNIX", address


def encode(
    op: str,
    fields: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> list[bytes]:
    """One message, as the frames to send in order.

    Returned rather than sent, so a value that cannot be sent is refused before the first frame
    leaves and the connection never holds half a message.
    """
    if op not in CLIENT_OPS and op not in REPLY_OPS:
        raise WireError(f"unknown op {op!r}")
    fields = {} if fields is None else fields
    if not isinstance(fields, Mapping) or not all(isinstance(k, str) for k in fields):
        raise WireError(f"{op}: fields must be a mapping with string keys")
    if arrays is None:
        arrays = {}
    if not isinstance(arrays, Mapping):
        raise WireError(f"{op}: arrays must be a mapping of names to arrays")
    described = []
    payloads = []
    for name in sorted(arrays, key=lambda n: (not isinstance(n, str), str(n))):
        if not isinstance(name, str) or not name:
            raise WireError(f"{op}: array name {name!r} is not a non-empty string")
        array = _checked(name, arrays[name])
        described.append({"name": name, "dtype": array.dtype.str, "shape": list(array.shape)})
        payloads.append(array.tobytes(order="C"))
    header = {"protocol": PROTOCOL_VERSION, "op": op, "fields": dict(fields), "arrays": described}
    try:
        encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise WireError(f"{op}: fields are not plain JSON: {exc}") from None
    if len(encoded) > MAX_HEADER_BYTES:
        raise WireError(f"{op}: header of {len(encoded)} bytes exceeds {MAX_HEADER_BYTES}")
    return [encoded, *payloads]


def _checked(name: str, value: Any) -> np.ndarray:
    array = value if isinstance(value, np.ndarray) else np.asarray(value)
    little = array.dtype.newbyteorder("<")
    if little.str not in DTYPES or array.dtype.hasobject:
        raise WireError(f"array {name!r}: dtype {array.dtype} cannot be sent")
    return array.astype(little, copy=False)


def send(
    conn: Any,
    op: str,
    fields: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> None:
    """Encode a message in full, then write its frames with `send_bytes`."""
    for frame in encode(op, fields, arrays):
        conn.send_bytes(frame)


def recv(
    conn: Any, *, max_bytes: int = MAX_MESSAGE_BYTES
) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """The next message: its op, its fields and its arrays, read with `recv_bytes` only.

    Raises `WireError` for a malformed message and lets `EOFError`/`OSError` through for a
    connection that is gone. After a `WireError` the frames that follow cannot be trusted to line
    up with messages, so the connection should be closed. Arrays are read-only.
    """
    raw = _frame(conn, MAX_HEADER_BYTES, "header")
    try:
        header = json.loads(raw.decode("utf-8"), parse_constant=_no_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise WireError(f"malformed header: {type(exc).__name__}: {exc}") from None
    if not isinstance(header, dict):
        raise WireError("header is not a JSON object")
    protocol = header.get("protocol")
    if type(protocol) is not int or protocol != PROTOCOL_VERSION:
        raise WireError(f"protocol {protocol!r}; this end speaks protocol {PROTOCOL_VERSION}")
    if set(header) != _HEADER_KEYS:
        raise WireError(f"header keys {sorted(header)}; expected {sorted(_HEADER_KEYS)}")
    op, fields, described = header["op"], header["fields"], header["arrays"]
    if not isinstance(op, str) or not op:
        raise WireError("header op is not a non-empty string")
    if not isinstance(fields, dict):
        raise WireError("header fields is not an object")
    if not isinstance(described, list):
        raise WireError("header arrays is not a list")

    specs: list[tuple[str, np.dtype, tuple[int, ...], int]] = []
    total = 0
    for entry in described:
        if not isinstance(entry, dict) or set(entry) != _ARRAY_KEYS:
            raise WireError(f"array description {entry!r} is not {{name, dtype, shape}}")
        name, dtype, shape = entry["name"], entry["dtype"], entry["shape"]
        if not isinstance(name, str) or not name:
            raise WireError(f"array name {name!r} is not a non-empty string")
        if any(name == seen for seen, *_ in specs):
            raise WireError(f"array {name!r} is described twice")
        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise WireError(f"array {name!r}: dtype {dtype!r} is not one this format carries")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise WireError(f"array {name!r}: shape {shape!r} is not a shape")
        nbytes = math.prod(shape) * np.dtype(dtype).itemsize
        total += nbytes
        if total > max_bytes:
            raise WireError(f"message arrays exceed {max_bytes} bytes at {name!r}")
        specs.append((name, np.dtype(dtype), tuple(shape), nbytes))

    arrays: dict[str, np.ndarray] = {}
    for name, dtype, shape, nbytes in specs:
        payload = _frame(conn, nbytes, f"array {name!r}")
        if len(payload) != nbytes:
            raise WireError(f"array {name!r}: {len(payload)} bytes for a {list(shape)} {dtype.str}")
        arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape)
    return op, fields, arrays


def _frame(conn: Any, limit: int, what: str) -> bytes:
    """One frame of at most `limit` bytes. A longer one is refused before it is allocated."""
    readable = getattr(conn, "readable", True)
    try:
        return conn.recv_bytes(max(limit, 1))
    except OSError:
        # `multiprocessing.connection` refuses an over-long frame by raising OSError and marking
        # the connection unreadable; a connection that was readable and is not any more is that.
        if readable and not getattr(conn, "readable", True):
            raise WireError(f"{what}: frame longer than {limit} bytes") from None
        raise


def _no_constant(token: str) -> Any:
    raise ValueError(f"{token} is not JSON")
