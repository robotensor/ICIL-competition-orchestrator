"""`icil-orchestrator store init|verify` and `queue add|list|remove`, through the console script."""

from __future__ import annotations

import json
import subprocess
import sysconfig
from pathlib import Path

from icil_orchestrator.canon import Signer
from icil_orchestrator.cli import main
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.queue import Queue
from icil_orchestrator.store.writer import Store
from store_helpers import TRACK, make_record, publish
from submission_helpers import SHA_A, FakeHub


def cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    script = Path(sysconfig.get_path("scripts")) / "icil-orchestrator"
    return subprocess.run([str(script), *args], capture_output=True, text=True, cwd=cwd)


def test_store_init_then_verify_passes(tmp_path):
    root = tmp_path / "store"
    init = cli("store", "init", str(root), cwd=tmp_path)
    assert init.returncode == 0, init.stderr
    manifest = json.loads(init.stdout)
    key = tmp_path / "keys" / "orchestrator.ed25519"
    assert "generated signing key" in init.stderr and key.stat().st_mode & 0o777 == 0o600
    assert manifest["validator_key"] == Signer.from_file(key).verify_key_hex
    assert (root / "tracks" / TRACK / "head.json").exists()

    verify = cli("store", "verify", str(root))
    assert verify.returncode == 0, verify.stdout
    assert verify.stdout.strip().splitlines()[-1] == "records=0 events=0 media=0 OK"


def test_verify_names_the_index_line_a_changed_byte_broke(spec, tmp_path):
    root, key = tmp_path / "store", tmp_path / "key"
    assert cli("store", "init", str(root), "--key", str(key)).returncode == 0
    store = Store(root, spec, Signer.from_file(key))
    king = SubmissionRef.make("org/genesis", "a" * 40)
    challenger = SubmissionRef.make("org/challenger", "b" * 40)
    publish(store, spec, make_record(spec, "genesis", 0, king, None))
    publish(store, spec, make_record(spec, "duel", 1, king, challenger))
    assert cli("store", "verify", str(root)).returncode == 0

    index = root / "tracks" / TRACK / "index-0000.jsonl"
    data = bytearray(index.read_bytes())
    second_line = data.index(b"\n") + 1
    at = data.index(b'"block":1', second_line) + len('"block":')
    data[at] = ord("8")
    index.write_bytes(bytes(data))

    verify = cli("store", "verify", str(root))
    assert verify.returncode == 1
    assert f"error: tracks/{TRACK}/index-0000.jsonl:2: bad signature" in verify.stdout
    assert verify.stdout.strip().endswith("FAILED")

    # A byte that is not UTF-8 is named the same way, not raised as a traceback.
    data[at] = 0xFF
    index.write_bytes(bytes(data))
    verify = cli("store", "verify", str(root))
    assert verify.returncode == 1 and "Traceback" not in verify.stderr, verify.stderr
    assert f"error: tracks/{TRACK}/index-0000.jsonl:2: not UTF-8" in verify.stdout


def test_verify_says_which_key_it_trusted_and_can_be_told_which_to_expect(spec, tmp_path):
    """manifest.json is unsigned, so a store re-signed under another key is self-consistent; the
    only way to know it is the competition's store is to pin the key."""
    root, key = tmp_path / "store", tmp_path / "key"
    assert cli("store", "init", str(root), "--key", str(key)).returncode == 0
    store = Store(root, spec, Signer.from_file(key))
    publish(
        store, spec, make_record(spec, "genesis", 0, SubmissionRef.make("org/g", "a" * 40), None)
    )
    mine = Signer.from_file(key).verify_key_hex

    ok = cli("store", "verify", str(root))
    assert ok.returncode == 0 and f"validator_key: {mine}" in ok.stdout
    pinned = cli("store", "verify", str(root), "--validator-key", mine)
    assert pinned.returncode == 0

    other = Signer.generate().verify_key_hex
    wrong = cli("store", "verify", str(root), "--validator-key", other)
    assert wrong.returncode == 1
    assert f"error: manifest.json is signed by {mine}, not the expected {other}" in wrong.stdout
    assert f"error: tracks/{TRACK}/index-0000.jsonl:1: bad signature" in wrong.stdout


def test_init_refuses_to_resign_a_store_with_another_key(tmp_path):
    root = tmp_path / "store"
    assert cli("store", "init", str(root), "--key", str(tmp_path / "a")).returncode == 0
    again = cli("store", "init", str(root), "--key", str(tmp_path / "a"))
    assert again.returncode == 0, "re-initialising with the same key is idempotent"
    assert (
        cli("store", "init", str(tmp_path / "other"), "--key", str(tmp_path / "b")).returncode == 0
    )
    other = cli("store", "init", str(root), "--key", str(tmp_path / "b"))
    assert other.returncode == 1 and "would invalidate every record" in other.stderr


def test_init_on_a_store_whose_key_is_missing_generates_nothing(tmp_path):
    """A key generated before the store was looked at is left lying in keys/, where the next run
    picks it up as the store's key."""
    root = tmp_path / "store"
    assert cli("store", "init", str(root), "--key", str(tmp_path / "a")).returncode == 0
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = cli("store", "init", str(root), cwd=elsewhere)
    assert done.returncode == 1 and "generated signing key" not in done.stderr
    assert "cannot be re-initialised" in done.stderr
    assert not (elsewhere / "keys").exists()


def test_init_refuses_a_key_inside_the_store(tmp_path):
    root = tmp_path / "store"
    done = cli("store", "init", str(root), "--key", str(root / "keys" / "k"))
    assert done.returncode == 2 and "outside the store" in done.stderr
    assert not (root / "keys").exists()


def test_queue_add_list_remove_and_publish_the_snapshot(spec, tmp_path, capsys):
    queue, root = tmp_path / "queue", tmp_path / "store"
    assert main(["store", "init", str(root), "--key", str(tmp_path / "k")]) == 0
    capsys.readouterr()
    base = ["queue", "--queue", str(queue), "--store", str(root)]

    assert main([*base, "add", "org/policy", "1" * 40, "--duel-size", "smoke"]) == 0
    added = capsys.readouterr().out
    key = SubmissionRef.make("org/policy", "1" * 40).key
    assert added.strip() == f"org/policy@{'1' * 40} key={key} track={TRACK} position=1"
    assert main([*base, "add", "org/other", "2" * 40]) == 0
    capsys.readouterr()

    assert main([*base, "list"]) == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[0].split()[:3] == ["1", key, f"org/policy@{'1' * 40}"]
    assert listing[-1] == "in_progress=- block=0"

    snapshot = json.loads((root / "tracks" / TRACK / "queue.json").read_text())
    assert [e["repo"] for e in snapshot["entries"]] == ["org/policy", "org/other"]

    assert main([*base, "remove", key]) == 0
    assert main([*base, "remove", key]) == 1
    capsys.readouterr()
    snapshot = json.loads((root / "tracks" / TRACK / "queue.json").read_text())
    assert [e["repo"] for e in snapshot["entries"]] == ["org/other"]
    assert main(["store", "verify", str(root)]) == 0


def test_queue_add_waits_for_no_one_while_a_duel_holds_the_store(spec, tmp_path, capsys):
    from icil_orchestrator.store.writer import store_lock

    queue, root = tmp_path / "queue", tmp_path / "store"
    assert main(["store", "init", str(root), "--key", str(tmp_path / "k")]) == 0
    capsys.readouterr()
    base = ["queue", "--queue", str(queue), "--store", str(root)]
    with store_lock(root):
        assert main([*base, "add", "org/policy", "1" * 40]) == 0
    out = capsys.readouterr()
    assert "it will publish the queue snapshot" in out.err
    assert not (root / "tracks" / TRACK / "queue.json").exists()
    assert main([*base, "list"]) == 0
    assert "org/policy" in capsys.readouterr().out, "the entry was still queued"


def test_queue_add_on_a_corrupt_queue_file_says_so(tmp_path, capsys):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / f"{TRACK}.json").write_text('{"entries": [], "block": 3')
    assert main(["queue", "--queue", str(queue), "add", "org/policy", "1" * 40]) == 2
    assert "is not a readable queue file" in capsys.readouterr().err


def test_queue_add_resolves_a_branch_or_tag_to_its_commit_once(tmp_path, capsys, monkeypatch):
    hub = FakeHub()
    hub.add("org/policy", SHA_A, {"icil.yaml": 80}, "main", "v1")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: hub)
    base = ["queue", "--queue", str(tmp_path / "queue")]
    assert main([*base, "add", "org/policy", "main"]) == 0
    out = capsys.readouterr()
    assert out.err.strip() == f"resolved org/policy@main to {SHA_A}"
    key = SubmissionRef.make("org/policy", SHA_A).key
    assert out.out.strip() == f"org/policy@{SHA_A} key={key} track={TRACK} position=1"
    (entry,) = Queue(tmp_path / "queue" / f"{TRACK}.json").entries()
    assert entry.revision == SHA_A, "the queue holds the commit, not the name"
    # The tag names the same commit: the same key, moved to the back, not queued twice.
    assert main([*base, "add", "org/policy", "v1"]) == 0
    assert [e.key for e in Queue(tmp_path / "queue" / f"{TRACK}.json").entries()] == [key]
    # A sha is queued as given, without asking the Hub.
    assert main([*base, "add", "org/other", "2" * 40]) == 0
    assert hub.calls == [("org/policy", "main"), ("org/policy", "v1")]


def test_queue_add_refuses_what_it_cannot_queue(tmp_path, capsys, monkeypatch):
    hub = FakeHub()
    hub.add("org/policy", SHA_A, {"icil.yaml": 80}, "main")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: hub)
    base = ["queue", "--queue", str(tmp_path / "queue")]
    assert main([*base, "add", "not a repo", "1" * 40]) == 2
    assert main([*base, "add", "org/policy", "1" * 40, "--duel-size", "enormous"]) == 2
    assert main(["queue", "--queue", str(tmp_path / "queue"), "--track", "video_only", "list"]) == 2
    # A name the Hub does not know, an abbreviation (another key for the same code) and a
    # repository that is not there are refused with the Hub's answer.
    assert main([*base, "add", "org/policy", "no-such-branch"]) == 2
    assert main([*base, "add", "org/policy", "a" * 7]) == 2
    assert main([*base, "add", "org/missing", "main"]) == 2
    err = capsys.readouterr().err
    assert "is not a Hugging Face repo id" in err and "is not one of smoke, light" in err
    assert "unknown track 'video_only'; the tracks are franka_1arm" in err
    assert err.count("Revision Not Found") == 2 and "Repository Not Found" in err
    assert hub.calls == [
        ("org/policy", "no-such-branch"),
        ("org/policy", "a" * 7),
        ("org/missing", "main"),
    ]
    assert (
        not list((tmp_path / "queue").glob("*.json"))
        or not Queue(tmp_path / "queue" / f"{TRACK}.json").entries()
    )
