from __future__ import annotations

import pytest

from icil_orchestrator.canon import Signer
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.store.records import unit_verdict_from_unit
from icil_orchestrator.store.verify import verify_store
from icil_orchestrator.store.writer import Store
from store_helpers import TRACK, make_record, publish, small_spec

KING = SubmissionRef.make("org/genesis", "a" * 40)
CHALLENGER = SubmissionRef.make("org/challenger", "b" * 40)
INDEX = f"tracks/{TRACK}/index-0000.jsonl"


@pytest.fixture
def history(spec, tmp_path):
    """Genesis, a defended duel and a dethroning duel, with one clip."""
    sp = small_spec(spec, tmp_path, lines_per_part=1000)
    store = Store(tmp_path / "store", sp, Signer.generate())
    store.init(store.signer.verify_key_hex)
    clip = tmp_path / "evaluation.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 100)
    sha = store.put_media(clip)
    unit = unit_verdict_from_unit(
        {
            "unit_id": "fp-000",
            "skill": "franka_pick_and_place",
            "index": 0,
            "task": "place_a2b_left",
            "instance": 0,
            "seed": 5,
            "instance_params": {
                "scene_seed": 9,
                "embodiment": ["franka-panda", "franka-panda", 0.6],
            },
            "demo": "fp-000",
        },
        view="sensorimotor",
    )
    unit.update(
        demo_video=sha,
        king_video=sha,
        king_success=True,
        challenger_success=False,
        outcome="king",
    )
    publish(store, sp, make_record(sp, "genesis", 0, KING, None))
    publish(store, sp, make_record(sp, "duel", 1, KING, CHALLENGER), units=[unit])
    publish(store, sp, make_record(sp, "duel", 2, KING, CHALLENGER, dethroned=True))
    return store, sp, sha


def test_a_published_history_verifies(history):
    store, sp, _ = history
    report = verify_store(store.root, sp)
    assert report.ok, report.errors
    # One clip referenced twice in the duel (demonstration and king) is one media file.
    assert (report.records, report.events, report.media) == (3, 3, 1)


def _flip_one_byte(store, line_no: int, old: bytes, new: bytes) -> None:
    path = store.root / INDEX
    lines = path.read_bytes().split(b"\n")
    assert len(old) == len(new) == 1 and lines[line_no - 1].count(old) >= 1
    lines[line_no - 1] = lines[line_no - 1].replace(old, new, 1)
    path.write_bytes(b"\n".join(lines))


def test_changing_one_byte_of_an_index_line_fails_and_names_the_line(history):
    store, sp, _ = history
    # A byte that keeps the line valid, canonical JSON, so only the signature can catch it: the
    # block number of the second record, 1 -> 7.
    path = store.root / INDEX
    lines = path.read_text().split("\n")
    assert '"block":1,' in lines[1]
    lines[1] = lines[1].replace('"block":1,', '"block":7,', 1)
    path.write_text("\n".join(lines))

    report = verify_store(store.root, sp)
    assert not report.ok
    assert f"{INDEX}:2: bad signature" in report.errors
    assert all(e.startswith(f"{INDEX}:2: ") for e in report.errors), report.errors


@pytest.mark.parametrize(
    "line_no, old, new, error",
    [
        (1, b"{", b"[", "unparsable record"),
        (3, b"\t", b" ", "no signature"),
        (2, b"t", b"T", "bad signature"),
    ],
)
def test_every_kind_of_one_byte_damage_is_named(history, line_no, old, new, error):
    store, sp, _ = history
    _flip_one_byte(store, line_no, old, new)
    report = verify_store(store.root, sp)
    assert f"{INDEX}:{line_no}: {error}" in report.errors, report.errors


def test_a_flipped_signature_byte_is_named(history):
    store, sp, _ = history
    path = store.root / INDEX
    lines = path.read_text().split("\n")
    body, sig = lines[0].split("\t")
    sig = ("1" if sig[0] != "1" else "2") + sig[1:]
    lines[0] = f"{body}\t{sig}"
    path.write_text("\n".join(lines))
    assert verify_store(store.root, sp).errors == [f"{INDEX}:1: bad signature"]


def test_missing_media_and_a_stale_head_are_errors(history):
    store, sp, sha = history
    store.media_path(sha, "mp4").unlink()
    store.write_head(TRACK, seq=2, event_id="x", block=0, finished_at="", king=None)
    errors = verify_store(store.root, sp).errors
    assert f"{INDEX}:2: media {sha[:12]} missing" in errors
    assert f"tracks/{TRACK}/head.json does not point at the last record" in errors


def test_the_schema_rejects_scores_outside_zero_to_one(spec, tmp_path):
    store = Store(tmp_path / "store", spec, Signer.generate())
    store.init(store.signer.verify_key_hex)
    record = make_record(spec, "duel", 1, KING, CHALLENGER)
    record["king_scores"] = {"franka_pick_and_place": 50, "average": 1.5}
    publish(store, spec, record)
    assert any("schema IndexRecord" in e for e in verify_store(store.root, spec).errors)


def test_a_store_that_does_not_exist_fails(spec, tmp_path):
    assert verify_store(tmp_path / "nothing", spec).errors == [
        "manifest.json missing or unreadable"
    ]
