"""The submission intake (`icil_orchestrator.admin`) over real HTTP, with the Hub stood in for."""

from __future__ import annotations

import http.client
import json
import logging
import socket

import httpx
import pytest

from icil_orchestrator.admin import MAX_BODY_BYTES, AdminServer
from icil_orchestrator.cli import main
from icil_orchestrator.ids import SubmissionRef, is_commit_sha
from icil_orchestrator.queue import Queue, Queues
from icil_orchestrator.store.writer import Store
from store_helpers import TRACK
from submission_helpers import SHA_A, SHA_B, FakeHub

TOKEN = "tok-3f9a1c7e-never-in-a-log"
DASHBOARD = {"track": TRACK, "duel_size": "smoke", "source": "dashboard-dev-mode"}


@pytest.fixture
def hub():
    fake = FakeHub()
    fake.add("org/policy", SHA_A, {"icil.yaml": 80}, "main", "v1")
    fake.add("org/other", SHA_B, {"icil.yaml": 80}, "main")
    return fake


@pytest.fixture
def paths(tmp_path):
    store, queue = tmp_path / "store", tmp_path / "queue"
    assert main(["store", "init", str(store), "--key", str(tmp_path / "key")]) == 0
    return store, queue


@pytest.fixture
def serve(spec, hub, paths):
    servers = []

    def start(**kwargs) -> AdminServer:
        store, queue = paths
        kwargs.setdefault("api", hub)
        server = AdminServer(
            spec,
            Queues(queue, spec.tracks),
            TOKEN,
            host="127.0.0.1",
            port=0,
            store=Store(store, spec),
            **kwargs,
        )
        server.start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()


@pytest.fixture
def server(serve):
    return serve()


def call(server, method, path, body=None, *, token=TOKEN, raw=None, headers=None):
    """One request on its own connection: `(status, JSON body, response headers)`."""
    conn = http.client.HTTPConnection(server.host, server.port, timeout=10)
    try:
        sent = {} if token is None else {"Authorization": f"Bearer {token}"}
        data = raw if raw is not None else None if body is None else json.dumps(body).encode()
        if data is not None:
            sent["Content-Type"] = "application/json"
        sent.update(headers or {})
        conn.request(method, path, body=data, headers=sent)
        response = conn.getresponse()
        return response.status, json.loads(response.read()), response
    finally:
        conn.close()


def headers_only(server, method, path, headers):
    """A request whose headers are sent and whose body never is, to see what is answered first."""
    conn = http.client.HTTPConnection(server.host, server.port, timeout=10)
    try:
        conn.putrequest(method, path, skip_accept_encoding=True)
        for name, value in headers.items():
            conn.putheader(name, value)
        conn.endheaders()
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def queued(paths):
    return Queue(paths[1] / f"{TRACK}.json").entries()


def test_health_answers_only_with_the_token(spec, server):
    status, body, response = call(server, "GET", "/admin/health", token=None)
    assert status == 401 and body["ok"] is False and body["error"]
    assert response.getheader("WWW-Authenticate").startswith("Bearer")
    assert call(server, "GET", "/admin/health", token="not-the-token")[0] == 401
    assert call(server, "GET", "/admin/health", token=TOKEN.upper())[0] == 401
    status, body, _ = call(server, "GET", "/admin/health")
    assert status == 200
    assert body == {
        "ok": True,
        "spec_version": spec.version,
        "tracks": list(spec.tracks),
        "queue_lengths": {TRACK: 0},
    }
    # Unauthenticated, nothing else is told apart: not the routes, not the methods.
    assert call(server, "GET", "/admin/elsewhere", token=None)[0] == 401
    assert call(server, "GET", "/admin/elsewhere")[0] == 404
    status, _, response = call(server, "GET", "/admin/submissions")
    assert status == 405 and response.getheader("Allow") == "POST"


def test_a_branch_is_queued_at_its_commit_and_listed(spec, server, hub, paths, capsys):
    status, body, _ = call(
        server,
        "POST",
        "/admin/submissions",
        {"repo": "org/policy", "revision": "main", **DASHBOARD},
    )
    assert status == 200, body
    key = SubmissionRef.make("org/policy", SHA_A).key
    assert is_commit_sha(body["revision"]) and body["revision"] == SHA_A
    assert body["ok"] is True and body["queued"] is True
    assert (body["key"], body["repo"], body["entry"]) == (key, "org/policy", f"org/policy@{SHA_A}")
    assert (body["position"], body["duel_size"], body["track"]) == (1, "smoke", TRACK)
    assert body["accepted_at"] and body["source"] == "dashboard-dev-mode" and body["message"]
    assert hub.calls == [("org/policy", "main")]

    (entry,) = queued(paths)
    assert (entry.key, entry.revision, entry.source) == (key, SHA_A, "dashboard-dev-mode")
    capsys.readouterr()
    assert main(["queue", "--queue", str(paths[1]), "list"]) == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[0].split()[:3] == ["1", key, f"org/policy@{SHA_A}"]
    snapshot = json.loads((paths[0] / "tracks" / TRACK / "queue.json").read_text())
    assert [e["key"] for e in snapshot["entries"]] == [key], "the snapshot was not published"
    assert call(server, "GET", "/admin/health")[1]["queue_lengths"] == {TRACK: 1}


def test_no_revision_is_the_default_branch_and_the_track_can_go_unsaid(server, hub, paths):
    status, body, _ = call(server, "POST", "/admin/submissions", {"repo": "org/other"})
    assert status == 200, body
    assert body["revision"] == SHA_B and body["duel_size"] is None and body["track"] == TRACK
    assert body["source"] == "admin"
    assert hub.calls == [("org/other", "main")]
    status, body, _ = call(
        server, "POST", "/admin/submissions", {"repo": "org/policy", "revision": None, **DASHBOARD}
    )
    assert status == 200 and body["revision"] == SHA_A and body["position"] == 2


def test_a_second_post_answers_with_the_place_it_already_has(server, hub, paths):
    first = call(server, "POST", "/admin/submissions", {"repo": "org/policy", **DASHBOARD})[1]
    call(server, "POST", "/admin/submissions", {"repo": "org/other", **DASHBOARD})
    # The same code, named by its commit this time: the same key, so the same submission.
    status, again, _ = call(
        server,
        "POST",
        "/admin/submissions",
        {"repo": "org/policy", "revision": SHA_A, "track": TRACK, "duel_size": "light"},
    )
    assert status == 200 and again["ok"] is True and again["queued"] is False
    assert (again["key"], again["position"]) == (first["key"], 1)
    assert (again["accepted_at"], again["duel_size"]) == (first["accepted_at"], "smoke")
    assert "nothing new was queued" in again["message"]
    assert [e.repo for e in queued(paths)] == ["org/policy", "org/other"]


@pytest.mark.parametrize(
    "body, status, field",
    [
        ({"repo": "org/policy", "revision": None, "track": "video_only"}, 422, "track"),
        ({"repo": "org/policy", "track": ["franka_1arm"]}, 422, "track"),
        ({"repo": "org/policy", "track": TRACK, "duel_size": "enormous"}, 422, "duel_size"),
        ({"repo": "org/policy", "track": TRACK, "skip_model_config_check": True}, 400, "body"),
        ({"repo": "org/policy", "tracks": [TRACK]}, 400, "body"),
        ({"repo": "not a repo"}, 422, "repo"),
        ({"repo": "org/policy", "revision": 7}, 422, "revision"),
        ({"repo": "org/policy", "revision": "has space"}, 422, "revision"),
        ({"repo": "org/policy", "source": "<script>"}, 422, "source"),
        ({}, 422, "repo"),
        ([{"repo": "org/policy"}], 400, "body"),
        ("org/policy", 400, "body"),
    ],
)
def test_what_is_not_a_submission_is_refused_before_the_hub(
    server, hub, paths, body, status, field
):
    got, answer, _ = call(server, "POST", "/admin/submissions", body)
    assert (got, answer["ok"], answer.get("field")) == (status, False, field), answer
    assert answer["error"]
    assert hub.calls == [] and queued(paths) == []


def test_bodies_that_are_not_json_objects_of_a_bounded_size_are_refused(server, hub, paths):
    too_big = json.dumps({"repo": "org/policy", "source": "x" * MAX_BODY_BYTES}).encode()
    status, body, response = call(server, "POST", "/admin/submissions", raw=too_big)
    assert status == 413 and body["field"] == "body" and str(MAX_BODY_BYTES) in body["error"]
    assert response.getheader("Connection") == "close"
    for raw in (b"{", b"\xff\xfe", b"not json", b'{"repo": NaN}', b"[" * 4000 + b"]" * 4000):
        status, body, _ = call(server, "POST", "/admin/submissions", raw=raw)
        assert (status, body["field"]) == (400, "body"), raw[:20]
    status, body, _ = call(
        server, "POST", "/admin/submissions", raw=b'{"repo": "org/policy", "repo": "org/other"}'
    )
    assert status == 400 and "twice" in body["error"]
    assert hub.calls == [] and queued(paths) == []


def test_the_length_is_checked_before_a_byte_of_the_body_is_read(server, hub, paths):
    auth = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    # Declared far past the limit and never sent: the refusal does not wait for it.
    status, body = headers_only(
        server, "POST", "/admin/submissions", {**auth, "Content-Length": "10000000"}
    )
    assert status == 413
    status, body = headers_only(
        server, "POST", "/admin/submissions", {**auth, "Transfer-Encoding": "chunked"}
    )
    assert status == 411 and "chunked" in body["error"]
    status, body = headers_only(server, "POST", "/admin/submissions", auth)
    assert status == 411 and "Content-Length" in body["error"]
    status, body = headers_only(
        server, "POST", "/admin/submissions", {**auth, "Content-Length": "-1"}
    )
    assert status == 400
    assert hub.calls == [] and queued(paths) == []


def test_a_body_that_does_not_arrive_in_time_is_refused(serve, paths):
    server = serve(request_timeout_s=0.3)
    status, body = headers_only(
        server,
        "POST",
        "/admin/submissions",
        {"Authorization": f"Bearer {TOKEN}", "Content-Length": "40"},
    )
    assert status == 408 and queued(paths) == []


def test_the_hub_refusing_is_422_with_its_reason_and_an_outage_503(server, hub, paths):
    status, body, _ = call(
        server, "POST", "/admin/submissions", {"repo": "org/policy", "revision": "no-such-branch"}
    )
    assert status == 422 and body["ok"] is False
    assert (
        body["error"]
        == "org/policy@no-such-branch: revision not found: Invalid rev id: no-such-branch"
    )
    status, body, _ = call(server, "POST", "/admin/submissions", {"repo": "org/missing"})
    assert status == 422 and body["error"] == "org/missing@main: repository not found"
    status, body, _ = call(
        server, "POST", "/admin/submissions", {"repo": "org/policy", "revision": "0" * 40}
    )
    assert status == 422 and "revision not found" in body["error"]

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("[Errno 101] Network is unreachable")

    hub.repo_info = unreachable
    status, body, _ = call(server, "POST", "/admin/submissions", {"repo": "org/policy"})
    assert status == 503 and "the Hub is unreachable" in body["error"]
    assert queued(paths) == []


def test_the_token_never_reaches_a_log(server, caplog, capfd):
    caplog.set_level(logging.DEBUG)
    call(server, "GET", "/admin/health")
    call(server, "GET", "/admin/health", token=f"{TOKEN}-but-longer")
    call(server, "GET", f"/admin/{TOKEN}?token={TOKEN}")
    call(server, "POST", f"/admin/submissions?t={TOKEN}", {"repo": "org/policy", **DASHBOARD})
    call(server, "POST", "/admin/submissions", {"repo": "org/policy", "note": TOKEN})
    call(server, "POST", "/admin/submissions", raw=b"{" + TOKEN.encode())
    # A request line that is not HTTP is answered and logged by http.server itself.
    with socket.create_connection((server.host, server.port), timeout=5) as sock:
        sock.sendall(f"BEARER {TOKEN} {TOKEN} {TOKEN}\r\n\r\n".encode())
        sock.recv(4096)
    out, err = capfd.readouterr()
    logged = caplog.text
    assert TOKEN not in logged and TOKEN not in out and TOKEN not in err
    key = SubmissionRef.make("org/policy", SHA_A).key
    (accepted,) = [
        r.getMessage() for r in caplog.records if "accepted submission" in r.getMessage()
    ]
    assert f"key={key}" in accepted and "repo=org/policy" in accepted
    assert f"sha={SHA_A}" in accepted and "source=dashboard-dev-mode" in accepted
    assert "[redacted]" in logged, "the request lines were not logged at all"


def test_no_token_no_server(spec, hub, paths):
    for token in ("", "   "):
        with pytest.raises(ValueError, match="does not serve without"):
            AdminServer(spec, Queues(paths[1], spec.tracks), token, port=0, api=hub)


@pytest.mark.network
def test_a_real_repository_resolves_through_the_hub(serve, paths):
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
    except OSError:
        pytest.skip("the Hugging Face Hub cannot be reached from here")
    server = serve(api=None)
    status, body, _ = call(
        server,
        "POST",
        "/admin/submissions",
        {"repo": "hf-internal-testing/tiny-random-bert", "revision": None, **DASHBOARD},
    )
    assert status == 200, body
    assert is_commit_sha(body["revision"]) and body["queued"] is True
    (entry,) = queued(paths)
    assert entry.revision == body["revision"]
