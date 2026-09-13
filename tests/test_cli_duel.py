"""`icil-orchestrator duel` and `daemon`, with the fake benchmark and the example policies."""

from __future__ import annotations

import json
import subprocess
import sysconfig
from pathlib import Path

import pytest

from duel_helpers import REPLAY, REPLAY_REF, ZERO, ZERO_REF
from icil_orchestrator.cli import main
from icil_orchestrator.duel.side import read_results
from icil_orchestrator.store.writer import Store
from submission_helpers import FakeHub

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


def run(spec, store, tmp_path, command: str, *args: str) -> int:
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
            "--key",
            str(key),
            "--runtime",
            "local",
            "--local",
            f"{REPLAY_REF.repo}={REPLAY}",
            "--local",
            f"{ZERO_REF.repo}={ZERO}",
            *args,
        ]
    )


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
    script = Path(sysconfig.get_path("scripts")) / "icil-orchestrator"
    for command in ("duel", "daemon"):
        done = subprocess.run([str(script), command, "--help"], capture_output=True, text=True)
        assert done.returncode == 0
        assert "WITHOUT A SANDBOX" in " ".join(done.stdout.split())
