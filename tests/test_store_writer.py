from __future__ import annotations

import json

import pytest

from icil_orchestrator.canon import Signer, canonical_json, verify_signature
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.store.records import unit_verdict_from_unit
from icil_orchestrator.store.writer import Store, store_lock
from store_helpers import TRACK, make_record, publish, small_spec

KING = SubmissionRef.make("org/genesis", "a" * 40)
CHALLENGER = SubmissionRef.make("org/challenger", "b" * 40)


def test_the_layout_the_dashboard_reads(spec, tmp_path):
    signer = Signer.generate()
    store = Store(tmp_path / "store", spec, signer)
    manifest = store.init(signer.verify_key_hex)
    assert manifest == {
        "schema": 4,
        "validator_key": signer.verify_key_hex,
        "spec_version": spec.version,
        "spec_fingerprint": spec.fingerprint,
        "tracks": [TRACK],
    }
    head = json.loads((tmp_path / "store" / "tracks" / TRACK / "head.json").read_text())
    assert head == {
        "schema": 4,
        "seq": 0,
        "event_id": "",
        "block": 0,
        "finished_at": "",
        "king": None,
    }

    record = make_record(spec, "genesis", 0, KING, None)
    publish(store, spec, record)
    line = (tmp_path / "store" / "tracks" / TRACK / "index-0000.jsonl").read_text()
    body, signature = line.rstrip("\n").split("\t")
    assert body == canonical_json({**record, "seq": 1})
    assert verify_signature(signer.verify_key_hex, body, signature)
    assert (tmp_path / "store" / "events" / TRACK / f"{record['event_id']}.json").exists()


def test_append_rotates_parts_and_moves_the_head(spec, tmp_path):
    sp = small_spec(spec, tmp_path)
    store = Store(tmp_path / "store", sp, Signer.generate())
    store.init(store.signer.verify_key_hex)
    assert publish(store, sp, make_record(sp, "genesis", 0, KING, None)) == 1
    assert store.head(TRACK)["king"]["key"] == KING.key
    assert publish(store, sp, make_record(sp, "duel", 1, KING, CHALLENGER)) == 2
    assert store.head(TRACK)["king"]["key"] == KING.key
    assert publish(store, sp, make_record(sp, "duel", 2, KING, CHALLENGER, dethroned=True)) == 3
    assert store.head(TRACK)["king"]["key"] == CHALLENGER.key
    assert store.index_part_path(TRACK, 1).exists()
    assert len(store.index_part_path(TRACK, 0).read_text().strip().split("\n")) == 2
    assert [r["seq"] for r in store.iter_index(TRACK)] == [1, 2, 3]


def test_a_torn_final_line_is_skipped_by_readers(spec, tmp_path):
    store = Store(tmp_path / "store", spec, Signer.generate())
    store.init(store.signer.verify_key_hex)
    publish(store, spec, make_record(spec, "genesis", 0, KING, None))
    with open(store.index_part_path(TRACK, 0), "a") as fh:
        fh.write('{"seq":2,"event_id":"ab')
    assert [r["seq"] for r in store.iter_index(TRACK)] == [1]


def test_media_is_content_addressed(spec, tmp_path):
    store = Store(tmp_path / "store", spec, Signer.generate())
    clip = tmp_path / "evaluation.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 100)
    sha = store.put_media(clip)
    assert store.has_media(sha, "mp4") and store.media_path(sha, "mp4").parent.name == sha[:2]
    assert store.put_media(clip) == sha


def test_a_foreign_kind_carrying_a_king_does_not_take_the_crown(spec, tmp_path):
    store = Store(tmp_path / "store", spec, Signer.generate())
    store.init(store.signer.verify_key_hex)
    publish(store, spec, make_record(spec, "genesis", 0, KING, None))
    usurper = SubmissionRef.make("org/usurper", "c" * 40)
    foreign = make_record(spec, "duel", 1, KING, usurper, dethroned=True)
    foreign["kind"] = "something_else"
    store.append(TRACK, foreign)
    head = store.head(TRACK)
    assert head["king"]["key"] == KING.key, "a non-crowning kind moved the crown"
    assert head["seq"] == 2


def test_a_unit_verdict_starts_unrun(spec):
    verdict = unit_verdict_from_unit(
        {
            "unit_id": "fp-000",
            "skill": "franka_pick_and_place",
            "index": 0,
            "task": "place_a2b_left",
            "task_label": "Place A to B (left)",
            "instance": 0,
            "seed": 5,
            "instance_params": {
                "scene_seed": 11,
                "embodiment": ["franka-panda", "franka-panda", 0.6],
            },
            "demo": "fp-000",
        },
        view="sensorimotor",
    )
    assert verdict["prompt"] == {
        "demo_id": "fp-000",
        "steps": 0,
        "chunks": 0,
        "view": "sensorimotor",
        "sha256": None,
    }
    assert verdict["void"] is False and verdict["king_success"] is None


def test_store_lock_is_exclusive(tmp_path):
    with store_lock(tmp_path / "s"):
        with pytest.raises(RuntimeError, match="another orchestrator"):
            with store_lock(tmp_path / "s"):
                pass
    with store_lock(tmp_path / "s"):
        pass
