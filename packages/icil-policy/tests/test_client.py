"""`RemotePolicy` against the real server in a subprocess, and against servers that misbehave."""

import json
import os
import pickle
import secrets
import shutil
import socket
import struct
import tempfile
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import pytest
from policy_testing import free_tcp_address, observation

from icil_policy import PolicyUnavailable, WireError, wire
from icil_policy.client import RemotePolicy


class Boom:
    """Unpickling this creates `path`: proof, if the file appears, that something unpickled it."""

    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (open, (self.path, "w"))


#: A key for servers that never check it.
KEY = bytes(range(32))


def remote(server, **kwargs):
    kwargs.setdefault("timeout_s", 30.0)
    kwargs.setdefault("log_file", server.log_file)
    return RemotePolicy(server.address, server.authkey, **kwargs)


# -- against the real server ----------------------------------------------------------------


@pytest.mark.parametrize("transport", ["unix", "tcp"])
def test_the_replay_example_gets_the_demonstrations_next_action_back_end_to_end(
    examples, serve, demonstration, transport
):
    address = free_tcp_address() if transport == "tcp" else None
    server = serve(examples / "replay_policy" / "icil.yaml", address=address)
    arrays, info = demonstration
    actions = arrays["actions"]
    with remote(server) as policy:
        hello = policy.hello()
        assert hello == {
            "protocol": 1,
            "action_type": "qpos",
            "policy": "replay.policy:ReplayPolicy",
        }
        assert policy.action_type == "qpos"
        policy.set_demonstration(arrays, info)
        policy.reset(0)
        for k in range(len(actions) + 2):
            reply = policy.act(observation(arrays, min(k, len(actions))))
            assert set(reply) == {"action"}
            assert reply["action"].dtype == np.float64 and reply["action"].shape == (16,)
            np.testing.assert_array_equal(reply["action"], actions[min(k, len(actions) - 1)])
        policy.reset(np.int64(1))
        np.testing.assert_array_equal(policy.act(observation(arrays))["action"], actions[0])
    assert server.wait() == 0


def test_the_zero_example_answers_zeros_of_an_action_row(examples, serve, demonstration):
    server = serve(examples / "zero_policy" / "icil.yaml")
    arrays, info = demonstration
    with remote(server) as policy:
        assert policy.hello()["action_type"] == "qpos"
        policy.set_demonstration(arrays, info)
        policy.reset(0)
        action = policy.act(observation(arrays))["action"]
        assert action.shape == (16,) and not action.any()
    assert server.wait() == 0


def test_a_call_that_exceeds_its_timeout_raises_and_the_server_exits(probe_repo, serve):
    server = serve(probe_repo(kwargs={"act_sleep_s": 3600}))
    policy = remote(server, timeout_s=1.0)
    policy.hello()
    started = time.monotonic()
    with pytest.raises(PolicyUnavailable) as caught:
        policy.act({"qpos": np.zeros(16)})
    assert time.monotonic() - started < 5
    assert caught.value.op == "act"
    assert "act: no answer within 1s" in str(caught.value)
    assert "serving probe:Probe" in caught.value.log_tail  # the server's log, quoted
    assert server.wait(timeout=10) == 0  # the server has exited, act still asleep inside it
    assert "hung up during act" in server.log()
    with pytest.raises(PolicyUnavailable, match="closed after an earlier failure"):
        policy.reset(0)
    policy.close()  # nothing left to close: no error


def test_a_policy_exception_raises_with_the_log_tail_and_the_policy_stays_usable(probe_repo, serve):
    server = serve(probe_repo())
    with remote(server) as policy:
        policy.hello()
        with pytest.raises(PolicyUnavailable) as caught:
            policy.act({"fail": np.array(True)})
        assert caught.value.remote_type == "RuntimeError"
        assert "act: RuntimeError: act: act failed on purpose" in str(caught.value)
        assert "probe: failing on purpose" in str(caught.value)
        assert "--- policy log (tail) ---" in str(caught.value)
        assert policy.act({"qpos": np.zeros(3)})["action"].shape == (7,)
    assert server.wait() == 0


def test_without_a_log_file_the_tail_the_server_sent_is_quoted(probe_repo, serve):
    server = serve(probe_repo())
    with remote(server, log_file=None) as policy:
        policy.hello()
        with pytest.raises(PolicyUnavailable, match="probe: failing on purpose"):
            policy.act({"fail": np.array(True)})


def test_a_policy_that_cannot_be_built_raises_on_hello(probe_repo, serve):
    server = serve(probe_repo(policy="broken:Policy"))
    policy = remote(server)
    with pytest.raises(PolicyUnavailable) as caught:
        policy.hello()
    assert caught.value.remote_type == "ImportError"
    assert "broken on purpose" in str(caught.value)
    assert server.wait() == 1
    with pytest.raises(PolicyUnavailable):
        policy.reset(0)
    policy.close()


def test_an_array_that_cannot_be_sent_is_the_callers_error_and_sends_nothing(probe_repo, serve):
    server = serve(probe_repo())
    with remote(server) as policy:
        policy.hello()
        with pytest.raises(WireError, match="cannot be sent"):
            policy.act({"qpos": np.array([object()], dtype=object)})
        with pytest.raises(WireError, match="cannot be sent"):
            policy.set_demonstration({"meta": np.array('{"scene_seed": 1}')}, {})
        with pytest.raises(WireError, match="cannot be sent"):
            policy.act({"qpos": [[1.0, 2.0], [3.0]]})
        with pytest.raises(TypeError):
            policy.reset(True)
        assert policy.act({"qpos": np.zeros(3)})["action"].shape == (7,)


def test_a_wrong_key_raises(probe_repo, serve):
    server = serve(probe_repo())
    with pytest.raises(PolicyUnavailable, match="refused this key"):
        RemotePolicy(server.address, secrets.token_bytes(32), timeout_s=10)


def test_nothing_listening_raises_once_the_timeout_has_passed(tmp_path):
    log_file = tmp_path / "serve.log"
    log_file.write_text("ModuleNotFoundError: no module named torch\n")
    directory = tempfile.mkdtemp(prefix="icilp-")
    started = time.monotonic()
    try:
        with pytest.raises(PolicyUnavailable) as caught:
            RemotePolicy(
                os.path.join(directory, "nobody.sock"), KEY, timeout_s=0.5, log_file=log_file
            )
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    assert 0.4 < time.monotonic() - started < 5
    assert "nothing listened" in str(caught.value)
    assert "no module named torch" in str(caught.value)


def test_a_server_that_starts_listening_late_is_waited_for(examples, serve):
    directory = tempfile.mkdtemp(prefix="icilp-")
    address = os.path.join(directory, "late.sock")
    key = secrets.token_bytes(32)
    connected = {}

    def connect():
        try:
            connected["policy"] = RemotePolicy(address, key, timeout_s=30)
        except PolicyUnavailable as exc:  # pragma: no cover - reported below
            connected["error"] = exc

    thread = threading.Thread(target=connect)
    thread.start()
    try:
        time.sleep(0.5)
        server = serve(examples / "zero_policy" / "icil.yaml", address=address, authkey=key)
        thread.join(timeout=30)
        assert "policy" in connected, connected
        with connected["policy"] as policy:
            assert policy.hello()["action_type"] == "qpos"
        assert server.wait() == 0
    finally:
        thread.join(timeout=30)
        shutil.rmtree(directory, ignore_errors=True)


def test_a_listener_that_never_authenticates_raises_within_the_timeout(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)  # connections queue in the backlog; nobody ever answers them
        address = f"127.0.0.1:{silent.getsockname()[1]}"
        started = time.monotonic()
        with pytest.raises(PolicyUnavailable, match="authentication did not finish within 0.5s"):
            RemotePolicy(address, KEY, timeout_s=0.5)
        assert time.monotonic() - started < 5


@pytest.mark.parametrize("challenge", [b"not a challenge", b""], ids=["garbage", "empty"])
def test_a_server_that_sends_a_garbage_challenge_raises_and_is_hung_up_on(challenge):
    directory = tempfile.mkdtemp(prefix="icilp-")
    address = os.path.join(directory, "garbage.sock")
    caught = {}

    def connect():
        try:
            RemotePolicy(address, secrets.token_bytes(32), timeout_s=5)
        except Exception as exc:  # which one is what the test checks
            caught["error"] = exc

    try:
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(address)
            listener.listen(1)
            thread = threading.Thread(target=connect)
            thread.start()
            conn, _ = listener.accept()
            with conn:
                conn.sendall(struct.pack("!i", len(challenge)) + challenge)
                thread.join(timeout=10)
                assert isinstance(caught.get("error"), PolicyUnavailable), caught
                assert caught["error"].op == "connect"
                conn.settimeout(5)
                while conn.recv(4096):  # times out if the client kept its end open
                    pass
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_arguments_are_checked_before_connecting():
    with pytest.raises(TypeError):
        RemotePolicy("/tmp/x.sock", "not bytes")
    with pytest.raises(ValueError):
        RemotePolicy("/tmp/x.sock", KEY, timeout_s=0)
    with pytest.raises(PolicyUnavailable, match="neither"):
        RemotePolicy("", KEY)
    with pytest.raises(ValueError, match="at least 16 bytes"):
        RemotePolicy("/tmp/x.sock", b"\x00" * 15)


# -- against servers that misbehave ---------------------------------------------------------


@pytest.fixture
def fake_server():
    """`fake_server(handler)`: a thread accepting one client and handing it to `handler(conn)`."""
    stop = threading.Event()
    cleanup = []

    def start(handler):
        directory = tempfile.mkdtemp(prefix="icilp-")
        address = os.path.join(directory, "fake.sock")
        authkey = secrets.token_bytes(16)
        listener = Listener(address, family="AF_UNIX", authkey=authkey)

        def run():
            conn = listener.accept()
            try:
                handler(conn)
                stop.wait(30)
            except (EOFError, OSError):
                pass
            finally:
                conn.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        cleanup.append((listener, thread, directory))
        return address, authkey

    yield start
    stop.set()
    for listener, thread, directory in cleanup:
        listener.close()
        thread.join(timeout=10)
        shutil.rmtree(directory, ignore_errors=True)


def answering(*frames_per_request):
    """A handler that answers the n-th request with the n-th list of raw frames."""

    def handler(conn):
        for frames in frames_per_request:
            wire.recv(conn)
            for frame in frames:
                conn.send_bytes(frame)

    return handler


HELLO_OK = wire.encode("ok", {"protocol": 1, "action_type": "ee", "policy": "fake:Fake"})


def test_a_pickled_reply_raises_and_is_never_unpickled(fake_server, tmp_path):
    marker = tmp_path / "unpickled"
    address, key = fake_server(answering(HELLO_OK, [pickle.dumps(Boom(marker))]))
    policy = RemotePolicy(address, key, timeout_s=10)
    policy.hello()
    with pytest.raises(PolicyUnavailable, match="malformed reply"):
        policy.act({"qpos": np.zeros(2)})
    assert not marker.exists()


def test_an_object_array_in_a_reply_raises(fake_server):
    head = {"protocol": 1, "op": "action", "fields": {}, "arrays": []}
    head["arrays"] = [{"name": "action", "dtype": "|O", "shape": [1]}]
    evil = [json.dumps(head).encode(), pickle.dumps(np.array([object()]))]
    address, key = fake_server(answering(HELLO_OK, evil))
    policy = RemotePolicy(address, key, timeout_s=10)
    policy.hello()
    with pytest.raises(PolicyUnavailable, match="dtype '\\|O'"):
        policy.act({"qpos": np.zeros(2)})


@pytest.mark.parametrize(
    ("hello", "act", "match"),
    [
        (wire.encode("ok", {"protocol": 2, "action_type": "ee"}), None, "speaks protocol 2"),
        (wire.encode("ok", {"protocol": 1, "action_type": "torque"}), None, "action_type"),
        (wire.encode("action", arrays={"action": np.zeros(1)}), None, "answered 'action'"),
        (HELLO_OK, wire.encode("ok"), "answered 'ok', not 'action'"),
        (HELLO_OK, wire.encode("action", arrays={"x": np.zeros(2)}), "no usable 'action'"),
        (HELLO_OK, wire.encode("action", arrays={"action": np.zeros(())}), "no usable"),
        (HELLO_OK, wire.encode("action", arrays={"action": np.zeros(0)}), r"shape \(0,\)"),
        (HELLO_OK, wire.encode("action", arrays={"action": np.zeros((0, 7))}), r"shape \(0, 7\)"),
        (HELLO_OK, wire.encode("action", arrays={"action": np.zeros((4, 0))}), r"shape \(4, 0\)"),
        (HELLO_OK, [b'{"protocol": 1, "op": "action"}'], "header keys"),
    ],
)
def test_a_reply_that_breaks_the_protocol_raises(fake_server, hello, act, match):
    address, key = fake_server(answering(hello, act or []))
    policy = RemotePolicy(address, key, timeout_s=10)
    with pytest.raises(PolicyUnavailable, match=match):
        policy.hello()
        policy.act({"qpos": np.zeros(2)})


def test_a_server_that_stalls_halfway_through_a_reply_raises_within_the_timeout(fake_server):
    head = {"protocol": 1, "op": "action", "fields": {}, "arrays": []}
    head["arrays"] = [{"name": "action", "dtype": "<f8", "shape": [2]}]
    address, key = fake_server(answering(HELLO_OK, [json.dumps(head).encode()]))
    policy = RemotePolicy(address, key, timeout_s=1)
    policy.hello()
    started = time.monotonic()
    with pytest.raises(PolicyUnavailable, match="no answer within 1s"):
        policy.act({"qpos": np.zeros(2)})
    assert time.monotonic() - started < 5


def test_a_server_that_never_reads_a_large_demonstration_raises_within_the_timeout(fake_server):
    address, key = fake_server(answering(HELLO_OK))  # answers hello, then never reads again
    policy = RemotePolicy(address, key, timeout_s=1)
    policy.hello()
    started = time.monotonic()
    with pytest.raises(PolicyUnavailable, match="no answer within 1s"):
        policy.set_demonstration({"frames_head": np.zeros((64, 1024, 1024), np.uint8)}, {})
    assert time.monotonic() - started < 5


def test_a_server_that_hangs_up_raises(fake_server):
    address, key = fake_server(lambda conn: (wire.recv(conn), conn.close()))
    policy = RemotePolicy(address, key, timeout_s=10)
    with pytest.raises(PolicyUnavailable, match="went away"):
        policy.hello()


def test_a_reply_claiming_more_than_a_reply_may_carry_raises_before_reading_it(fake_server):
    head = {"protocol": 1, "op": "action", "fields": {}, "arrays": []}
    head["arrays"] = [{"name": "action", "dtype": "<f8", "shape": [1 << 40]}]
    address, key = fake_server(answering(HELLO_OK, [json.dumps(head).encode()]))
    policy = RemotePolicy(address, key, timeout_s=10)
    policy.hello()
    with pytest.raises(PolicyUnavailable, match="exceed"):
        policy.act({"qpos": np.zeros(2)})


def action_header(shape):
    head = {"protocol": 1, "op": "action", "fields": {}}
    head["arrays"] = [{"name": "action", "dtype": "<f8", "shape": shape}]
    return json.dumps(head).encode()


@pytest.mark.parametrize(
    "reply",
    [
        [action_header([1] * 70), b"\0" * 8],
        [action_header([0, 2**64]), b""],
        [action_header([0, 2**63]), b""],
        [b"[" * 200_000],
    ],
    ids=["ndim-70", "dim-2**64", "dim-2**63", "nested-json"],
)
def test_a_reply_numpy_cannot_build_raises_and_abandons_the_connection(fake_server, reply):
    address, key = fake_server(answering(HELLO_OK, reply))
    policy = RemotePolicy(address, key, timeout_s=10)
    policy.hello()
    with pytest.raises(PolicyUnavailable, match="malformed reply"):
        policy.act({"qpos": np.zeros(2)})
    with pytest.raises(PolicyUnavailable, match="closed after an earlier failure"):
        policy.reset(0)


def test_anything_else_a_reply_raises_while_being_read_is_policy_unavailable(
    fake_server, monkeypatch
):
    real_recv = wire.recv

    def recv(conn, **kwargs):  # the fake server's thread keeps the real one
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("an unforeseen way to fail")
        return real_recv(conn, **kwargs)

    address, key = fake_server(answering(HELLO_OK))
    policy = RemotePolicy(address, key, timeout_s=10)
    monkeypatch.setattr(wire, "recv", recv)
    with pytest.raises(PolicyUnavailable, match="malformed reply: RuntimeError: an unforeseen"):
        policy.hello()
    with pytest.raises(PolicyUnavailable, match="closed after an earlier failure"):
        policy.reset(0)


def test_a_reply_describing_too_many_arrays_raises_within_the_timeout(fake_server):
    head = {"protocol": 1, "op": "ok", "fields": {"protocol": 1, "action_type": "ee"}}
    head["arrays"] = [{"name": f"a{i}", "dtype": "|u1", "shape": [0]} for i in range(80_000)]
    address, key = fake_server(answering([json.dumps(head).encode()]))
    policy = RemotePolicy(address, key, timeout_s=1)
    started = time.monotonic()
    with pytest.raises(PolicyUnavailable, match="malformed reply"):
        policy.hello()
    assert time.monotonic() - started < 5


def test_the_context_manager_does_not_hide_the_exception_that_ended_the_block(fake_server):
    address, key = fake_server(answering(HELLO_OK))
    with pytest.raises(KeyError), RemotePolicy(address, key, timeout_s=1) as policy:
        policy.hello()
        raise KeyError("the benchmark's own error")
