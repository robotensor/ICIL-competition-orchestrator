"""`python -m icil_policy.serve`, run as a real subprocess and spoken to over the raw wire format."""

import json
import pickle
import time
from multiprocessing import AuthenticationError

import numpy as np
import pytest
from policy_testing import AUTHKEY_ENV, call, free_tcp_address, observation, write_repo

from icil_policy import wire
from icil_policy.serve import checked_action


class Boom:
    """Unpickling this creates `path`: proof, if the file appears, that something unpickled it."""

    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (open, (self.path, "w"))


def mode(name):
    return np.frombuffer(name.encode(), np.uint8)


def state_of(reply):
    op, _, arrays = reply
    assert op == "action", reply
    return json.loads(bytes(arrays["state"]))


@pytest.mark.parametrize("transport", ["unix", "tcp"])
def test_the_replay_example_answers_hello_prompt_reset_act_with_the_next_action(
    examples, serve, demonstration, transport
):
    address = free_tcp_address() if transport == "tcp" else None
    server = serve(examples / "replay_policy" / "icil.yaml", address=address)
    arrays, info = demonstration
    conn = server.connect()

    op, fields, _ = call(conn, "hello", {"client": "test"})
    assert (op, fields) == (
        "ok",
        {"protocol": 1, "action_type": "qpos", "policy": "replay.policy:ReplayPolicy"},
    )
    assert call(conn, "prompt", {"info": info}, arrays)[0] == "ok"
    assert call(conn, "reset", {"seed": 3})[0] == "ok"
    actions = arrays["actions"]
    for k in range(len(actions) + 2):
        op, _, reply = call(conn, "act", {}, observation(arrays, min(k, len(actions))))
        assert op == "action"
        np.testing.assert_array_equal(reply["action"], actions[min(k, len(actions) - 1)])
    assert call(conn, "close")[0] == "ok"
    assert server.wait() == 0


def test_the_policy_is_imported_from_the_repository_root_without_the_key(
    probe_repo, serve, demonstration
):
    manifest = probe_repo()
    server = serve(manifest)
    arrays, info = demonstration
    conn = server.connect()
    assert call(conn, "hello", {"client": "test"})[1]["action_type"] == "ee"
    call(conn, "prompt", {"info": info}, arrays)
    call(conn, "reset", {"seed": 11})
    state = state_of(call(conn, "act", {}, observation(arrays)))
    assert state["cwd"] == str(manifest.parent)
    assert state["path0"] == str(manifest.parent)
    assert state["sibling"] == "from the repository root"
    assert state["key_in_env"] is False
    assert state["seed"] == 11
    assert state["info"] == info
    assert state["demo"] == {k: [v.dtype.str, list(v.shape)] for k, v in arrays.items()}
    assert state["writeable"] is False
    assert "probe: built" in server.log()


def test_a_policy_exception_is_an_error_reply_and_the_server_keeps_serving(probe_repo, serve):
    server = serve(probe_repo())
    conn = server.connect()
    call(conn, "hello")
    op, fields, _ = call(conn, "act", {}, {"fail": np.array(True)})
    assert op == "error"
    assert fields["type"] == "RuntimeError"
    assert "act failed on purpose" in fields["message"]
    assert "probe: failing on purpose" in fields["log_tail"]
    assert "Traceback" in fields["log_tail"]
    op, fields, _ = call(conn, "reset", {"seed": -1})
    assert (op, fields["type"]) == ("error", "ValueError")
    assert call(conn, "act", {}, {"qpos": np.zeros(3)})[0] == "action"
    assert call(conn, "close")[0] == "ok"
    assert server.wait() == 0


@pytest.mark.parametrize(
    ("returns", "kind", "message"),
    [
        ("none", "TypeError", "not a mapping"),
        ("no_action", "ValueError", "no 'action'"),
        ("scalar", "ValueError", "shape ()"),
        ("object", "WireError", "cannot be sent"),
    ],
)
def test_an_unusable_act_result_is_an_error_reply(probe_repo, serve, returns, kind, message):
    server = serve(probe_repo())
    conn = server.connect()
    call(conn, "hello")
    op, fields, _ = call(conn, "act", {}, {"returns": mode(returns)})
    assert (op, fields["type"]) == ("error", kind)
    assert message in fields["message"]
    op, _, arrays = call(conn, "act", {}, {"returns": mode("chunk")})
    assert op == "action" and arrays["action"].shape == (4, 7)
    op, _, arrays = call(conn, "act", {}, {"returns": mode("list")})
    assert op == "action" and arrays["action"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("op", "fields", "message"),
    [
        ("reset", {}, "seed must be an integer"),
        ("reset", {"seed": "3"}, "seed must be an integer"),
        ("reset", {"seed": True}, "seed must be an integer"),
        ("prompt", {"info": [1]}, "info must be an object"),
        ("ok", {}, "unknown op 'ok'"),
    ],
)
def test_a_well_formed_request_with_bad_fields_is_refused_and_serving_goes_on(
    probe_repo, serve, op, fields, message
):
    server = serve(probe_repo())
    conn = server.connect()
    call(conn, "hello")
    reply_op, reply, _ = call(conn, op, fields)
    assert (reply_op, reply["type"]) == ("error", "WireError")
    assert message in reply["message"]
    assert call(conn, "reset", {"seed": 1})[0] == "ok"


def test_an_unknown_op_in_a_well_formed_header_is_refused(probe_repo, serve):
    server = serve(probe_repo())
    conn = server.connect()
    call(conn, "hello")
    conn.send_bytes(json.dumps({"protocol": 1, "op": "exec", "fields": {}, "arrays": []}).encode())
    assert conn.poll(20)
    op, fields, _ = wire.recv(conn)
    assert op == "error" and "unknown op 'exec'" in fields["message"]
    assert call(conn, "reset", {"seed": 1})[0] == "ok"


def test_nothing_but_hello_or_close_is_served_before_hello(probe_repo, serve):
    server = serve(probe_repo())
    conn = server.connect()
    for op, fields in (("reset", {"seed": 1}), ("prompt", {}), ("act", {})):
        reply_op, reply, _ = call(conn, op, fields)
        assert reply_op == "error" and reply["message"] == f"{op} before hello"
    assert "probe: built" not in server.log()
    assert call(conn, "hello")[0] == "ok"
    assert call(conn, "reset", {"seed": 1})[0] == "ok"


@pytest.mark.parametrize("after_hello", [False, True])
def test_a_pickle_frame_gets_an_error_reply_and_is_never_unpickled(
    probe_repo, serve, tmp_path, after_hello
):
    marker = tmp_path / "unpickled"
    server = serve(probe_repo())
    conn = server.connect()
    if after_hello:
        call(conn, "hello")
    conn.send(Boom(marker))  # Connection.send: exactly what a pickling client would write
    assert conn.poll(20)
    op, fields, _ = wire.recv(conn)
    assert op == "error"
    assert fields["type"] == "WireError"
    assert "malformed header" in fields["message"]
    assert server.wait() == 1  # the session is over: nothing after a bad frame lines up
    assert not marker.exists()
    assert "malformed message" in server.log()


def test_an_object_array_announced_to_the_server_is_refused(probe_repo, serve):
    server = serve(probe_repo())
    conn = server.connect()
    call(conn, "hello")
    head = {
        "protocol": 1,
        "op": "act",
        "fields": {},
        "arrays": [{"name": "qpos", "dtype": "|O", "shape": [2]}],
    }
    conn.send_bytes(json.dumps(head).encode())
    conn.send_bytes(pickle.dumps(np.array([1, 2], dtype=object)))
    assert conn.poll(20)
    op, fields, _ = wire.recv(conn)
    assert op == "error" and "dtype '|O'" in fields["message"]
    assert server.wait() == 1


def test_a_client_that_hangs_up_during_a_call_that_never_returns_takes_the_server_with_it(
    probe_repo, serve
):
    server = serve(probe_repo(kwargs={"act_sleep_s": 3600}))
    conn = server.connect()
    call(conn, "hello")
    wire.send(conn, "act", {}, {"qpos": np.zeros(2)})
    time.sleep(0.5)
    assert server.process.poll() is None  # still inside act
    conn.close()
    assert server.wait(timeout=10) == 0
    assert "hung up during act" in server.log()


def test_a_client_that_hangs_up_between_calls_ends_the_server(probe_repo, serve):
    server = serve(probe_repo())
    conn = server.connect()
    call(conn, "hello")
    conn.close()
    assert server.wait() == 0


def test_close_calls_the_policy_close_and_exits(probe_repo, serve, tmp_path):
    closed = tmp_path / "closed"
    server = serve(probe_repo(kwargs={"close_marker": str(closed)}))
    conn = server.connect()
    call(conn, "hello")
    assert call(conn, "close")[0] == "ok"
    assert server.wait() == 0
    assert closed.read_text() == "closed"


def test_the_server_exits_after_close_even_if_the_policy_left_a_thread_running(probe_repo, serve):
    server = serve(probe_repo(kwargs={"linger": True}))
    conn = server.connect()
    call(conn, "hello")
    assert call(conn, "close")[0] == "ok"
    assert server.wait(timeout=10) == 0


@pytest.mark.parametrize(
    ("policy", "kwargs", "kind", "message"),
    [
        ("broken:Policy", None, "ImportError", "broken on purpose"),
        ("missing.module:Policy", None, "ModuleNotFoundError", "missing"),
        ("probe:Absent", None, "AttributeError", "'Absent'"),
        ("probe:Probe", {"broken_init": True}, "RuntimeError", "refuses to be built"),
        ("probe:Probe", {"unexpected": 1}, "TypeError", "unexpected"),
        ("notapolicy:NotAPolicy", None, "TypeError", "action_type is 'torque'"),
    ],
)
def test_a_policy_that_cannot_be_built_fails_hello_with_its_log_and_exits_1(
    probe_repo, serve, policy, kwargs, kind, message
):
    server = serve(probe_repo(policy=policy, kwargs=kwargs))
    conn = server.connect()
    op, fields, _ = call(conn, "hello", {"client": "test"})
    assert (op, fields["type"]) == ("error", kind)
    assert message in fields["message"]
    assert "hello raised" in fields["log_tail"]
    assert server.wait() == 1


def test_a_client_with_the_wrong_key_is_refused_and_the_right_one_is_still_served(
    probe_repo, serve
):
    server = serve(probe_repo())
    with pytest.raises(AuthenticationError):
        server.connect(authkey=b"not the key")
    conn = server.connect()
    assert call(conn, "hello")[0] == "ok"


@pytest.mark.parametrize(
    ("env", "manifest_text", "message"),
    [
        ({AUTHKEY_ENV: ""}, None, "does not hold a hex authkey"),
        ({AUTHKEY_ENV: "not hex"}, None, "does not hold a hex authkey"),
        ({}, "api: 2\npolicy: probe:Probe\n", "api: must be 1"),
    ],
)
def test_serving_that_cannot_start_exits_2_with_the_reason_in_the_log(
    tmp_path, serve, env, manifest_text, message
):
    manifest = write_repo(tmp_path / "repo")
    if manifest_text:
        manifest.write_text(manifest_text)
    server = serve(manifest, env=env, listening=False)
    assert server.wait() == 2
    assert message in server.log()


def test_checked_action_accepts_one_action_or_a_chunk_and_nothing_else():
    assert checked_action({"action": np.zeros(3)})["action"].shape == (3,)
    assert checked_action({"action": np.zeros((2, 3)), "aux": np.ones(1)})["aux"].shape == (1,)
    for bad in (None, [np.zeros(3)], {"action": np.zeros((1, 2, 3))}, {"action": np.zeros(0)}):
        with pytest.raises((TypeError, ValueError)):
            checked_action(bad)
