"""`vector-orchestrator duel` and `daemon`, with the fake benchmark and the example policies."""

from __future__ import annotations

import http.client
import json
import logging
import os
import signal
import socket
import subprocess
import sysconfig
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from duel_helpers import REPLAY, REPLAY_REF, ZERO, ZERO_REF
from submission_helpers import FakeHub
from vector_orchestrator.cli import main
from vector_orchestrator.duel.side import read_results
from vector_orchestrator.queue import Queues
from vector_orchestrator.store.writer import Store, store_lock

TRACK = "franka_1arm"


@pytest.fixture
def store(duel_spec, tmp_path, fake_installed):
    root, key = tmp_path / "store", tmp_path / "keys" / "orchestrator.ed25519"
    assert main(["--spec", str(duel_spec.path), "store", "init", str(root), "--key", str(key)]) == 0
    return root, key


@pytest.fixture
def hub(monkeypatch):
    """The Hub `queue add` confirms a commit with: both examples, at their refs' commits."""
    fake = FakeHub()
    for ref in (ZERO_REF, REPLAY_REF):
        fake.add(ref.repo, ref.revision, {"icil.yaml": 80})
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: fake)
    return fake


def run(spec, store, tmp_path, command: str, *args: str, zero: Path = ZERO) -> int:
    root, key = store
    return main(
        [
            "--spec",
            str(spec.path),
            command,
            "--store",
            str(root),
            "--run-dir",
            str(tmp_path / "runs"),
            "--queue",
            str(tmp_path / "queue"),
            "--key",
            str(key),
            "--runtime",
            "local",
            "--local",
            f"{REPLAY_REF.repo}={REPLAY}",
            "--local",
            f"{ZERO_REF.repo}={zero}",
            *args,
        ]
    )


ZERO_AT = f"{ZERO_REF.repo}@{ZERO_REF.revision}"
REPLAY_AT = f"{REPLAY_REF.repo}@{REPLAY_REF.revision}"


def test_a_duel_by_hand_is_numbered_from_the_queue_and_never_shares_the_daemons_run(
    duel_spec, store, hub, tmp_path, capsys
):
    assert run(duel_spec, store, tmp_path, "duel", "--challenger", ZERO_AT) == 0
    capsys.readouterr()
    # The king's directory is missing: the duel fails for the harness, undecided, at block 2.
    code = run(duel_spec, store, tmp_path, "duel", "--challenger", REPLAY_AT, zero=tmp_path / "no")
    assert code == 2 and "fetching the king" in capsys.readouterr().err
    by_hand = [d for d in (tmp_path / "runs" / TRACK).iterdir() if (d / "failed.txt").is_file()]
    assert len(by_hand) == 1 and Queues(tmp_path / "queue", duel_spec.tracks)[TRACK].block == 2

    queue = ["--spec", str(duel_spec.path), "queue", "--queue", str(tmp_path / "queue")]
    assert main([*queue, "add", REPLAY_REF.repo, REPLAY_REF.revision]) == 0
    assert run(duel_spec, store, tmp_path, "daemon", "--once") == 0
    records = Store(store[0], duel_spec).iter_index(TRACK)
    assert [(r["kind"], r["block"]) for r in records] == [("genesis", 1), ("duel", 3)]
    assert records[1]["event_id"][:16] != by_hand[0].name, "the daemon ran in the hand duel's run"
    assert (by_hand[0] / "failed.txt").is_file()


def test_a_failed_duel_by_hand_run_again_resumes_at_its_block(duel_spec, store, tmp_path, capsys):
    assert run(duel_spec, store, tmp_path, "duel", "--challenger", ZERO_AT) == 0
    capsys.readouterr()
    code = run(duel_spec, store, tmp_path, "duel", "--challenger", REPLAY_AT, zero=tmp_path / "no")
    assert code == 2
    capsys.readouterr()
    assert run(duel_spec, store, tmp_path, "duel", "--challenger", REPLAY_AT) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "published" and not (Path(out["run_dir"]) / "failed.txt").exists()
    assert len(list((tmp_path / "runs" / TRACK).iterdir())) == 2, "the failed duel was not resumed"
    assert Queues(tmp_path / "queue", duel_spec.tracks)[TRACK].block == 2


def test_the_duel_command_leaves_an_empty_throne_to_the_declared_baseline(
    spec_doc, write_spec, store, tmp_path, capsys
):
    from conftest import fake_spec_doc

    doc = fake_spec_doc(spec_doc)
    doc["budgets"]["act_timeout_s"] = 2.0
    doc["baselines"][TRACK] = {"repo": ZERO_REF.repo, "revision": ZERO_REF.revision}
    spec = write_spec(doc, name="baseline.json")
    assert run(spec, store, tmp_path, "duel", "--challenger", REPLAY_AT) == 2
    assert f"track {TRACK} declares a baseline" in capsys.readouterr().err
    assert Store(store[0], spec).iter_index(TRACK) == []
    assert not (tmp_path / "runs" / TRACK).exists()


def test_the_duel_command_refuses_while_a_daemon_holds_the_store(
    duel_spec, store, tmp_path, capsys
):
    with store_lock(store[0]):
        assert run(duel_spec, store, tmp_path, "duel", "--challenger", ZERO_AT) == 2
    assert "another orchestrator is publishing" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
    assert Queues(tmp_path / "queue", duel_spec.tracks)[TRACK].block == 0


def test_a_smoke_duel_publishes_every_unit_with_one_prompt_hash_for_both_sides(
    duel_spec, store, tmp_path, capsys
):
    """Acceptance: genesis on the empty track, then `duel --size smoke` publishes an event whose
    units carry the prompt hash both sides ran from, and `store verify` passes."""
    root, key = store
    zero = f"{ZERO_REF.repo}@{ZERO_REF.revision}"
    assert run(duel_spec, store, tmp_path, "duel", "--track", TRACK, "--challenger", zero) == 0
    genesis = json.loads(capsys.readouterr().out)
    assert genesis["status"] == "published" and genesis["kind"] == "genesis"

    replay = f"{REPLAY_REF.repo}@{REPLAY_REF.revision}"
    code = run(
        duel_spec,
        store,
        tmp_path,
        "duel",
        "--track",
        TRACK,
        "--challenger",
        replay,
        "--size",
        "smoke",
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "published" and out["kind"] == "duel" and out["dethroned"] is True
    assert out["new_king"] == REPLAY_REF.as_dict() and out["units"] == 3 and out["void"] == 0

    event = Store(root, duel_spec).event(TRACK, out["event_id"])
    run_dir = Path(out["run_dir"])
    challenger, king = read_results(run_dir / "challenger"), read_results(run_dir / "king")
    assert len(event["units"]) == 3
    for unit in event["units"]:
        sha = unit["prompt_sha256"]
        assert sha and challenger[unit["unit_id"]]["prompt_sha256"] == sha
        assert king[unit["unit_id"]]["prompt_sha256"] == sha

    validator = Store(root, duel_spec).manifest()["validator_key"]
    verify = ["--spec", str(duel_spec.path), "store", "verify", str(root)]
    assert main([*verify, "--validator-key", validator]) == 0
    # Clips counted per record: the genesis's demonstration and rollout, the duel's three each.
    assert capsys.readouterr().out.strip().endswith("records=2 events=2 media=15 OK")


def test_the_daemon_serves_the_queue_once_per_step(duel_spec, store, hub, tmp_path, capsys):
    root, _ = store
    queue = tmp_path / "queue"
    for ref in (ZERO_REF, REPLAY_REF):
        added = main(
            [
                "--spec",
                str(duel_spec.path),
                "queue",
                "--queue",
                str(queue),
                "add",
                ref.repo,
                ref.revision,
            ]
        )
        assert added == 0
    for _ in range(2):
        assert run(duel_spec, store, tmp_path, "daemon", "--queue", str(queue), "--once") == 0
    head = Store(root, duel_spec).head(TRACK)
    assert head["king"] == REPLAY_REF.as_dict() and head["block"] == 2
    snapshot = json.loads((root / "tracks" / TRACK / "queue.json").read_text())
    assert snapshot["entries"] == [] and snapshot["in_progress"] is None


ADMIN_TOKEN = "tok-daemon-7c2e91d04b5a-never-in-a-log"
ADMIN = ["--admin-token-env", "VECTOR_TEST_ADMIN_TOKEN"]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def closed(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    except ConnectionRefusedError:
        return True
    return False


def admin_call(port: int, method: str, path: str, body: Any = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_the_daemon_takes_what_its_intake_queues(
    duel_spec, store, hub, tmp_path, monkeypatch, capsys, caplog
):
    """`daemon --admin`: a submission posted to the intake goes on the daemon's own queue, the loop
    takes it and crowns it on the empty track, and a signal stops the intake with the loop."""
    root, _ = store
    caplog.set_level(logging.INFO, logger="vector_orchestrator.admin")
    hub.add(REPLAY_REF.repo, REPLAY_REF.revision, {"icil.yaml": 80}, "main")
    monkeypatch.setattr("vector_orchestrator.admin.HubApi", lambda: hub)
    monkeypatch.setenv("VECTOR_TEST_ADMIN_TOKEN", ADMIN_TOKEN)
    port = free_port()
    queue_file = tmp_path / "queue" / f"{TRACK}.json"
    seen: dict[str, Any] = {}
    returned = threading.Event()

    def submit_then_stop() -> None:
        deadline = time.monotonic() + 120
        try:
            while not returned.is_set() and time.monotonic() < deadline:
                try:
                    seen["health"] = admin_call(port, "GET", "/admin/health")
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                return
            submission = {"repo": REPLAY_REF.repo, "revision": "main", "track": TRACK}
            seen["submitted"] = admin_call(port, "POST", "/admin/submissions", submission)
            while not returned.is_set() and time.monotonic() < deadline:
                king = (Store(root, duel_spec).head(TRACK) or {}).get("king")
                state = json.loads(queue_file.read_text()) if queue_file.exists() else {}
                if king and not state.get("entries") and not state.get("in_progress"):
                    seen["king"] = king
                    return
                time.sleep(0.05)
        finally:
            if not returned.is_set():
                # SIGINT rather than SIGTERM: were the daemon gone, it interrupts the test run
                # instead of killing it.
                os.kill(os.getpid(), signal.SIGINT)

    helper = threading.Thread(target=submit_then_stop, daemon=True)
    helper.start()
    try:
        code = run(
            duel_spec,
            store,
            tmp_path,
            "daemon",
            "--idle-sleep",
            "0.05",
            "--admin",
            "--admin-port",
            str(port),
            *ADMIN,
        )
    finally:
        returned.set()
        helper.join(10)
    err = capsys.readouterr().err
    assert code == 128 + signal.SIGINT, err
    assert seen["health"] == (
        200,
        {
            "ok": True,
            "spec_version": duel_spec.version,
            "tracks": [TRACK],
            "queue_lengths": {TRACK: 0},
        },
    )
    status, body = seen["submitted"]
    assert status == 200 and body["queued"] and body["position"] == 1, body
    assert body["revision"] == REPLAY_REF.revision and body["key"] == REPLAY_REF.key
    assert seen.get("king") == REPLAY_REF.as_dict(), "the daemon did not take the entry"
    assert Store(root, duel_spec).head(TRACK)["block"] == 1
    snapshot = json.loads((root / "tracks" / TRACK / "queue.json").read_text())
    assert snapshot["entries"] == [] and snapshot["in_progress"] is None
    assert closed(port), "the intake outlived its daemon"
    assert f"admin intake listening on http://127.0.0.1:{port}" in caplog.text
    assert f"accepted submission key={REPLAY_REF.key}" in caplog.text


def test_the_daemons_intake_needs_its_token_and_stops_with_the_daemon(
    duel_spec, store, tmp_path, monkeypatch, capsys
):
    root, _ = store
    port = free_port()
    daemon_once = ["daemon", "--once", "--admin", "--admin-port", str(port), *ADMIN]
    monkeypatch.delenv("VECTOR_TEST_ADMIN_TOKEN", raising=False)
    assert run(duel_spec, store, tmp_path, *daemon_once) == 2
    assert "VECTOR_TEST_ADMIN_TOKEN is not set" in capsys.readouterr().err
    monkeypatch.setenv("VECTOR_TEST_ADMIN_TOKEN", "dev-token")
    assert run(duel_spec, store, tmp_path, *daemon_once) == 2
    err = capsys.readouterr().err
    assert "must be 32 or more" in err and "dev-token" not in err

    monkeypatch.setenv("VECTOR_TEST_ADMIN_TOKEN", ADMIN_TOKEN)
    assert run(duel_spec, store, tmp_path, *daemon_once) == 0
    assert closed(port), "the intake outlived a daemon that ran its round"
    with store_lock(root):
        assert run(duel_spec, store, tmp_path, *daemon_once) == 1
    assert "another orchestrator is publishing" in capsys.readouterr().err
    assert closed(port), "the intake outlived a daemon that could not take its store"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))  # and its port is free again


def test_the_daemons_intake_defaults_are_the_intake_defaults():
    from vector_orchestrator import admin
    from vector_orchestrator.cli import build_parser

    args = build_parser().parse_args(["daemon", "--store", "s", "--run-dir", "r"])
    assert args.admin is False
    assert (args.admin_host, args.admin_port, args.admin_token_env) == (
        admin.DEFAULT_HOST,
        admin.DEFAULT_PORT,
        admin.DEFAULT_TOKEN_ENV,
    )


def test_what_the_duel_command_refuses(duel_spec, store, tmp_path, capsys):
    root, key = store
    replay = f"{REPLAY_REF.repo}@{REPLAY_REF.revision}"
    assert run(duel_spec, store, tmp_path, "duel", "--challenger", replay, "--size", "huge") == 2
    assert "duel size 'huge' is not one of" in capsys.readouterr().err
    assert run(duel_spec, store, tmp_path, "duel", "--challenger", "org/nobody@main") == 2
    assert "no local directory for org/nobody" in capsys.readouterr().err
    bare = main(
        [
            "--spec",
            str(duel_spec.path),
            "duel",
            "--challenger",
            replay,
            "--store",
            str(root),
            "--run-dir",
            str(tmp_path / "runs"),
            "--key",
            str(key),
            "--runtime",
            "local",
        ]
    )
    assert bare == 2 and "--runtime local serves only what --local" in capsys.readouterr().err
    assert (
        run(duel_spec, store, tmp_path, "duel", "--challenger", replay, "--live-url", "http://x")
        == 2
    )
    assert "--live-token-env" in capsys.readouterr().err


def test_the_help_says_the_local_runtime_has_no_sandbox():
    script = Path(sysconfig.get_path("scripts")) / "vector-orchestrator"
    for command in ("duel", "daemon"):
        done = subprocess.run([str(script), command, "--help"], capture_output=True, text=True)
        assert done.returncode == 0
        assert "WITHOUT A SANDBOX" in " ".join(done.stdout.split())
