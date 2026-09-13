from __future__ import annotations

import json

import pytest

from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.queue import Queue, Queues
from icil_orchestrator.spec import load_schema
from icil_orchestrator.store.verify import Report, _Schema


def test_queue_replace_moves_to_back_and_persists(tmp_path):
    q = Queue(tmp_path / "q.json")
    e1, p1 = q.add("a/x", "1" * 40)
    e2, p2 = q.add("b/y", "2" * 40, duel_size="light")
    assert (p1, p2) == (1, 2)
    e1b, p1b = q.add("a/x", "1" * 40)
    assert p1b == 2 and e1b.key == e1.key and [e.key for e in q.entries()] == [e2.key, e1.key]
    q2 = Queue(tmp_path / "q.json")
    assert [e.key for e in q2.entries()] == [e2.key, e1.key]
    assert q2.pop().key == e2.key
    q2.start("e" * 64, SubmissionRef.make("b/y", "2" * 40))
    snap = q2.snapshot("franka_1arm", None, 4)
    assert snap["in_progress"]["event_id"] == "e" * 64 and snap["entries"][0]["position"] == 1
    q2.finish()
    assert Queue(tmp_path / "q.json").state.in_progress is None
    assert q2.remove(e1.key) and not q2.remove(e1.key)


def test_the_snapshot_is_the_schema_4_shape_the_dashboard_reads(tmp_path):
    q = Queue(tmp_path / "q.json")
    q.add("org/policy", "3" * 40, duel_size="smoke", now="2026-09-13T10:00:00Z")
    q.start("f" * 64, SubmissionRef.make("org/other", "4" * 40), now="2026-09-13T10:05:00Z")
    king = SubmissionRef.make("org/king", "5" * 40)
    snap = q.snapshot("franka_1arm", king, 4, now="2026-09-13T10:06:00Z")
    report = Report()
    _Schema(load_schema()).check("QueueSnapshot", snap, "queue.json", report)
    assert report.errors == []
    assert snap["entries"][0] == {
        "position": 1,
        "key": SubmissionRef.make("org/policy", "3" * 40).key,
        "repo": "org/policy",
        "revision": "3" * 40,
        "commit_block": 0,
        "duel_size": "smoke",
        "skip_model_config_check": False,
        "accepted_at": "2026-09-13T10:00:00Z",
    }
    assert snap["written_at"] == "2026-09-13T10:06:00Z" and snap["king"] == king.as_dict()


def test_a_queue_written_by_the_validator_still_loads(tmp_path):
    """Its entries carry `skip_model_config_check`, which this orchestrator no longer has."""
    path = tmp_path / "franka_1arm.json"
    entry = {
        "key": "k" * 16,
        "repo": "org/old",
        "revision": "6" * 40,
        "commit_block": 3,
        "duel_size": None,
        "skip_model_config_check": True,
        "accepted_at": "2026-01-01T00:00:00Z",
    }
    path.write_text(json.dumps({"entries": [entry], "in_progress": None, "block": 3}))
    (loaded,) = Queue(path).entries()
    assert loaded.repo == "org/old" and loaded.commit_block == 3


def test_one_queue_per_track(tmp_path):
    queues = Queues(tmp_path / "queue", ["franka_1arm"])
    queues["franka_1arm"].add("org/x", "7" * 40)
    assert (tmp_path / "queue" / "franka_1arm.json").exists()
    with pytest.raises(KeyError, match="the tracks are franka_1arm"):
        queues["other"]
    (tmp_path / "file").write_text("{}")
    with pytest.raises(ValueError, match="is a file"):
        Queues(tmp_path / "file", ["franka_1arm"])
