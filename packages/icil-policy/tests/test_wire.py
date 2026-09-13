import contextlib
import json
import pickle
import threading
import time
from multiprocessing import Pipe

import numpy as np
import pytest

from icil_policy import wire
from icil_policy.errors import WireError


class FrameConn:
    """A connection that only has `send_bytes` and `recv_bytes`: pickling calls would blow up."""

    def __init__(self, frames=()):
        self.frames = list(frames)

    def send_bytes(self, frame):
        self.frames.append(bytes(frame))

    def recv_bytes(self, maxlength=None):
        if not self.frames:
            raise EOFError
        frame = self.frames.pop(0)
        if maxlength is not None and len(frame) > maxlength:
            raise AssertionError("the fake does not model over-long frames")
        return frame

    def send(self, _obj):  # pragma: no cover - the point is that it is never called
        raise AssertionError("send pickles")

    def recv(self):  # pragma: no cover
        raise AssertionError("recv unpickles")


class Unconvertible:
    """Like a tensor on a GPU: it has a shape, and numpy cannot make an array of it."""

    shape = (7,)

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")


def nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


def header(**overrides):
    base = {"protocol": 1, "op": "act", "fields": {}, "arrays": []}
    base.update(overrides)
    return json.dumps(base).encode()


class Boom:
    """Unpickling this creates `path`: proof, if the file appears, that something unpickled it."""

    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (open, (self.path, "w"))


@pytest.mark.parametrize("dtype", sorted(wire.DTYPES))
def test_every_allowed_dtype_round_trips_through_a_real_connection(dtype):
    a, b = Pipe()
    value = (np.arange(24) % 2).astype(dtype).reshape(2, 3, 4)
    wire.send(a, "act", {"t": 3}, {"x": value})
    op, fields, arrays = wire.recv(b)
    assert (op, fields) == ("act", {"t": 3})
    assert arrays["x"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(arrays["x"], value)


def test_scalar_empty_non_contiguous_and_big_endian_arrays_keep_their_values():
    conn = FrameConn()
    big = np.arange(6, dtype=">f8").reshape(2, 3)
    sent = {
        "frequency": np.array(15.0),
        "empty": np.zeros((0, 16)),
        "strided": np.arange(20, dtype=np.int32).reshape(4, 5)[::2, ::-2],
        "big": big,
        "frames_head": np.full((2, 4, 5, 3), 7, dtype=np.uint8),
    }
    wire.send(conn, "prompt", {"info": {"frequency": 15.0}}, sent)
    op, fields, arrays = wire.recv(conn)
    assert op == "prompt" and fields == {"info": {"frequency": 15.0}}
    assert set(arrays) == set(sent)
    for name, value in sent.items():
        assert arrays[name].shape == value.shape
        np.testing.assert_array_equal(arrays[name], value)
    assert arrays["frequency"].shape == ()
    assert arrays["big"].dtype.str == "<f8"
    assert not arrays["big"].flags.writeable


def test_a_message_is_one_json_header_then_one_raw_frame_per_array_in_header_order():
    frames = wire.encode("act", {}, {"b": np.zeros(2, np.float32), "a": np.ones(3, np.uint8)})
    head = json.loads(frames[0])
    assert head["protocol"] == wire.PROTOCOL_VERSION
    assert [d["name"] for d in head["arrays"]] == ["a", "b"]
    assert head["arrays"][0] == {"name": "a", "dtype": "|u1", "shape": [3]}
    assert frames[1:] == [bytes([1, 1, 1]), np.zeros(2, "<f4").tobytes()]


@pytest.mark.parametrize(
    "value",
    [
        np.array([object(), 1], dtype=object),
        np.array([1.0, 2.0], dtype=object),
        ["a", 1],
        np.array(["text"]),
        np.array([1 + 2j]),
        np.array(["2026-09-13"], dtype="datetime64[D]"),
        np.zeros(2, dtype=[("x", "<f8")]),
    ],
    ids=["object", "object-floats", "mixed-list", "str", "complex", "datetime", "structured"],
)
def test_an_array_of_any_other_dtype_is_refused_before_anything_is_sent(value):
    conn = FrameConn()
    with pytest.raises(WireError, match="cannot be sent"):
        wire.send(conn, "act", {}, {"ok": np.zeros(1), "bad": value})
    assert conn.frames == []


STRING_DTYPE = getattr(getattr(np, "dtypes", None), "StringDType", None)  # numpy 2 only


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: [[1, 2], [3]], id="ragged"),
        pytest.param(Unconvertible, id="tensor-on-a-gpu"),
        pytest.param(
            lambda: np.array(["a"], dtype=STRING_DTYPE()),
            id="stringdtype",
            marks=pytest.mark.skipif(STRING_DTYPE is None, reason="numpy 1 has no StringDType"),
        ),
    ],
)
def test_a_value_numpy_cannot_turn_into_a_sendable_array_is_refused_before_anything_is_sent(make):
    conn = FrameConn()
    # numpy 1 makes an object array of a ragged list where numpy 2 raises; either way it is refused.
    with pytest.raises(WireError, match="'bad'.*cannot be sent"):
        wire.send(conn, "act", {}, {"ok": np.zeros(1), "bad": make()})
    assert conn.frames == []


def test_an_object_dtype_announced_by_a_header_is_refused_on_receive():
    payload = pickle.dumps(np.array([1, 2], dtype=object))
    conn = FrameConn(
        [
            header(arrays=[{"name": "x", "dtype": "|O", "shape": [2]}]),
            payload,
        ]
    )
    with pytest.raises(WireError, match="dtype '\\|O'"):
        wire.recv(conn)
    assert conn.frames == [payload]  # refused on the header, before its frame was read


@pytest.mark.parametrize("dtype", ["<O", "|O8", ">f8", "<c16", "<U4", "|V8", "<M8[s]", "f8"])
def test_any_dtype_outside_the_list_is_refused_on_receive(dtype):
    conn = FrameConn([header(arrays=[{"name": "x", "dtype": dtype, "shape": [1]}]), b"\0" * 16])
    with pytest.raises(WireError, match="dtype"):
        wire.recv(conn)


def test_a_pickle_frame_is_refused_and_never_unpickled(tmp_path):
    marker = tmp_path / "unpickled"
    evil = pickle.dumps(Boom(marker))
    with pytest.raises(FileNotFoundError):  # the payload works: loading it would create the file
        pickle.loads(pickle.dumps(Boom(tmp_path / "no-such-dir" / "x")))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises(WireError, match="malformed header"):
            wire.recv(FrameConn([pickle.dumps(Boom(marker), protocol=protocol)]))
    a, b = Pipe()
    a.send(Boom(marker))  # what Connection.send would put on the socket
    with pytest.raises(WireError, match="malformed header"):
        wire.recv(b)
    a.send_bytes(evil)
    with pytest.raises(WireError):
        wire.recv(b)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        ([b"[1, 2]"], "not a JSON object"),
        ([b'{"protocol": 1, "op": "act", "fields": {}, "arrays": [], "x": NaN}'], "malformed"),
        ([header(protocol=2)], "protocol 2"),
        ([header(protocol=True)], "protocol True"),
        ([header(protocol=1.0)], "protocol 1.0"),
        ([header(extra=1)], "header keys"),
        ([json.dumps({"protocol": 1, "op": "act", "fields": {}}).encode()], "header keys"),
        ([header(op="")], "op"),
        ([header(op=3)], "op"),
        ([header(fields=[])], "fields"),
        ([header(arrays={})], "arrays"),
        ([header(arrays=[{"name": "x", "dtype": "<f8"}])], "is not"),
        ([header(arrays=[{"name": "", "dtype": "<f8", "shape": [1]}])], "name"),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [-1]}])], "shape"),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [True]}])], "shape"),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": 3}])], "shape"),
        (
            [header(arrays=[{"name": "x", "dtype": "<f8", "shape": [1]}] * 2), b"\0" * 8] * 2,
            "twice",
        ),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [2]}]), b"\0" * 8], "8 bytes"),
        ([b"[" * 200_000], "malformed header"),
        ([b'{"protocol": 1, "fields": ' + b'{"a": ' * 100_000], "malformed header"),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [1] * 70}]), b"\0" * 8], "70 dim"),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [1] * 33}]), b"\0" * 8], "33 dim"),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [0, 2**64]}]), b""], "exceeds"),
        ([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [0, 2**63]}]), b""], "exceeds"),
        (
            [header(arrays=[{"name": "x", "dtype": "<f8", "shape": [0, 1 << 40, 1 << 40]}])],
            "exceeds",
        ),
    ],
)
def test_a_malformed_message_is_refused(frames, match):
    with pytest.raises(WireError, match=match):
        wire.recv(FrameConn(frames))


def test_the_most_dimensions_any_numpy_allows_are_carried():
    shape = [1] * wire.MAX_NDIM
    frames = [header(arrays=[{"name": "x", "dtype": "<f8", "shape": shape}]), b"\0" * 8]
    assert wire.recv(FrameConn(frames))[2]["x"].shape == tuple(shape)


@pytest.mark.parametrize(
    "frame",
    [
        header(protocol="p" * 1_000_000),
        header(op=["o"] * 500_000),
        header(arrays=[["e" * 1_000_000]]),
        header(arrays=[{"name": ["n"] * 500_000, "dtype": "<f8", "shape": [1]}]),
        header(arrays=[{"name": "x", "dtype": "d" * 1_000_000, "shape": [1]}]),
        header(arrays=[{"name": "x", "dtype": "<f8", "shape": [[[[[[[[1]]]]]]]] * 100_000}]),
    ],
    ids=["protocol", "op", "entry", "name", "dtype", "shape"],
)
def test_a_refusal_quotes_no_more_than_an_excerpt_of_what_it_refuses(frame):
    with pytest.raises(WireError) as caught:
        wire.recv(FrameConn([frame]))
    assert len(str(caught.value)) < 500


def test_a_header_claiming_more_than_the_receiver_accepts_is_refused_before_any_frame():
    huge = header(arrays=[{"name": "x", "dtype": "<f8", "shape": [1 << 40, 1 << 20]}])
    conn = FrameConn([huge, b"never read"])
    with pytest.raises(WireError, match="exceed"):
        wire.recv(conn)
    assert conn.frames == [b"never read"]
    small = FrameConn([header(arrays=[{"name": "x", "dtype": "<f8", "shape": [4]}]), b"\0" * 32])
    with pytest.raises(WireError, match="exceed"):
        wire.recv(small, max_bytes=16)


def test_a_header_describing_more_arrays_than_a_message_may_hold_is_refused_at_once():
    # Zero-size arrays cost no bytes, so only a count bounds how long checking the header takes.
    many = [{"name": f"a{i}", "dtype": "|b1", "shape": [0]} for i in range(80_000)]
    frame = header(arrays=many)
    assert len(frame) < wire.MAX_HEADER_BYTES
    started = time.monotonic()
    with pytest.raises(WireError, match=f"at most {wire.MAX_ARRAYS}"):
        wire.recv(FrameConn([frame]))
    assert time.monotonic() - started < 2
    most = [{"name": f"a{i}", "dtype": "|b1", "shape": [0]} for i in range(wire.MAX_ARRAYS)]
    started = time.monotonic()
    assert len(wire.recv(FrameConn([header(arrays=most), *[b""] * wire.MAX_ARRAYS]))[2]) == len(
        most
    )
    assert time.monotonic() - started < 2
    with pytest.raises(WireError, match=f"at most {wire.MAX_ARRAYS}"):
        wire.encode("act", arrays={f"a{i}": np.zeros(0) for i in range(wire.MAX_ARRAYS + 1)})


def test_a_frame_longer_than_its_description_is_refused_without_reading_it():
    a, b = Pipe()
    a.send_bytes(header(arrays=[{"name": "x", "dtype": "<f8", "shape": [1]}]))
    a.send_bytes(b"\0" * 4096)
    with pytest.raises(WireError, match="longer than 8 bytes"):
        wire.recv(b)
    c, d = Pipe()

    # Only the length prefix is ever read, so the frame's body need not fit in the socket buffer.
    def oversized():
        with contextlib.suppress(OSError):
            c.send_bytes(b"\0" * (wire.MAX_HEADER_BYTES + 1))

    sender = threading.Thread(target=oversized, daemon=True)
    sender.start()
    with pytest.raises(WireError, match="header: frame longer"):
        wire.recv(d)
    d.close()
    sender.join(timeout=10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"op": "exec"}, "unknown op"),
        ({"op": "ok", "fields": {"x": float("nan")}}, "plain JSON"),
        ({"op": "ok", "fields": {"x": np.int64(1)}}, "plain JSON"),
        ({"op": "ok", "fields": {1: "x"}}, "string keys"),
        ({"op": "ok", "fields": ["x"]}, "string keys"),
        ({"op": "act", "arrays": [np.zeros(1)]}, "mapping"),
        ({"op": "act", "arrays": {"": np.zeros(1)}}, "name"),
        ({"op": "act", "arrays": {3: np.zeros(1)}}, "name"),
        ({"op": "ok", "fields": {"x": nested(100_000)}}, "plain JSON"),
    ],
)
def test_a_message_that_cannot_be_encoded_is_refused(kwargs, match):
    with pytest.raises(WireError, match=match):
        wire.encode(**kwargs)


@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < "2.0.0", reason="numpy 1 stops at 32")
def test_an_array_with_more_dimensions_than_numpy_1_allows_is_refused_on_send():
    with pytest.raises(WireError, match="33 dimensions"):
        wire.encode("act", arrays={"x": np.zeros((1,) * 33)})


def test_the_eof_of_a_closed_connection_is_not_a_wire_error():
    a, b = Pipe()
    a.close()
    with pytest.raises(EOFError):
        wire.recv(b)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("/tmp/icil/policy.sock", ("AF_UNIX", "/tmp/icil/policy.sock")),
        ("policy.sock", ("AF_UNIX", "policy.sock")),
        ("127.0.0.1:5555", ("AF_INET", ("127.0.0.1", 5555))),
        ("localhost:80", ("AF_INET", ("localhost", 80))),
        ("./run/a:1", ("AF_UNIX", "./run/a:1")),
    ],
)
def test_an_address_is_a_socket_path_or_host_and_port(address, expected):
    assert wire.parse_address(address) == expected


@pytest.mark.parametrize("address", ["", "host:0", "host:70000"])
def test_an_unusable_address_is_refused(address):
    with pytest.raises(WireError):
        wire.parse_address(address)
