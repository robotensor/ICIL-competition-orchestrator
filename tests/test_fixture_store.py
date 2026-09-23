"""The committed fixture store is what the orchestrator writes today, verifies, and reads the way
the dashboard's `lib/store.ts` (robotensor/robotensor-competition-dashboard, milestone-two-contests)
reads a store."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from vector_orchestrator.store.verify import verify_store

FIXTURES = Path(__file__).resolve().parent / "fixtures"
STORE = FIXTURES / "store"
EVENT_ID = re.compile(r"^[0-9a-f]{8,64}$")


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_fixture_store_is_what_the_generator_writes_today(tmp_path):
    out = tmp_path / "store"
    subprocess.run([sys.executable, str(FIXTURES / "make_store.py"), str(out)], check=True)
    assert tree(out) == tree(STORE), (
        "the fixture store is stale: if the change was intended, run "
        "`python tests/fixtures/make_store.py` and commit tests/fixtures/store"
    )


def test_the_fixture_store_verifies_against_the_shipped_spec(spec):
    report = verify_store(STORE, spec)
    assert report.ok, report.errors
    assert report.warnings == []
    assert (report.records, report.events) == (2, 2)


def test_the_dashboard_can_read_the_fixture_store(spec):
    track = "franka_1arm"
    manifest = json.loads((STORE / "manifest.json").read_text())
    assert manifest["tracks"] == [track] and manifest["schema"] == spec.store["schema"]

    head = json.loads((STORE / "tracks" / track / "head.json").read_text())
    # indexRecords: parts are probed from 0 up to floor(seq / per) + 1.
    per = spec.store["index_lines_per_part"]
    assert isinstance(head["seq"], int) and head["seq"] // per + 1 >= 1

    records = []
    for line in (STORE / "tracks" / track / "index-0000.jsonl").read_text().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line[: line.index("\t")])
        # parseIndexPart keeps a record only with a numeric seq and string event_id and kind.
        assert isinstance(record["seq"], int) and isinstance(record["event_id"], str)
        assert isinstance(record["kind"], str)
        records.append(record)
    assert [r["kind"] for r in records] == ["genesis", "duel"]

    duel = records[-1]
    assert duel["dethroned"] and duel["new_king"]["repo"] == "robotensor/vector-replay-policy"
    assert head["king"] == duel["new_king"]
    for record in records:
        assert EVENT_ID.match(record["event_id"])
        event = json.loads((STORE / "events" / track / f"{record['event_id']}.json").read_text())
        assert isinstance(event["units"], list)
        # A unit whose skill the spec does not name is silently dropped by the dashboard.
        assert {u["skill"] for u in event["units"]} <= set(spec.skills(track))
    duel_event = json.loads((STORE / "events" / track / f"{duel['event_id']}.json").read_text())
    assert len(duel_event["units"]) == spec.units_per_side(track, "light")
    assert sum(u["void"] for u in duel_event["units"]) == duel["void"] == 1

    queue = json.loads((STORE / "tracks" / track / "queue.json").read_text())
    assert [(e["repo"], e["position"]) for e in queue["entries"]] == [
        ("robotensor/vector-example-policy", 1)
    ]
