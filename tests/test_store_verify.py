from __future__ import annotations

import json

import pytest

from icil_orchestrator.canon import Signer, canonical_json
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.store.records import duel_event, unit_verdict_from_unit
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


@pytest.mark.parametrize(
    "damage, error",
    [
        # One byte that is not UTF-8 at all.
        (lambda line: line.replace(b'"block":1', b'"block":\xff', 1), "not UTF-8"),
        # JSON parsers accept NaN; a signed record cannot hold it.
        (lambda line: line.replace(b'"block":1', b'"block":NaN', 1), "unparsable record"),
        # Deep enough to exhaust the parser's recursion.
        (lambda line: b"[" * 200_000 + line, "unparsable record"),
    ],
)
def test_bytes_no_parser_expects_are_named_not_raised(history, damage, error):
    store, sp, _ = history
    path = store.root / INDEX
    lines = path.read_bytes().split(b"\n")
    lines[1] = damage(lines[1])
    path.write_bytes(b"\n".join(lines))
    report = verify_store(store.root, sp)
    assert f"{INDEX}:2: {error}" in report.errors, report.errors


@pytest.mark.parametrize(
    "damage, error",
    [
        # Upper-casing a hex digit leaves the signature bytes the same, but not the line.
        (
            lambda data: data.replace(
                data.split(b"\t", 1)[1][:128], data.split(b"\t", 1)[1][:128].upper(), 1
            ),
            f"{INDEX}:1: signature is not 128 lowercase hex characters",
        ),
        (
            lambda data: data[:-1] + b" ",
            f"{INDEX}:3: signature is not 128 lowercase hex characters",
        ),
        (
            lambda data: data[:-1] + b"\r",
            f"{INDEX}:3: signature is not 128 lowercase hex characters",
        ),
        (lambda data: data[:-1], f"{INDEX}:3: no newline at the end of the line"),
        (lambda data: data.replace(b"\n", b"\n\n", 1), f"{INDEX}:2: blank line"),
    ],
)
def test_a_byte_that_leaves_the_signed_content_alone_still_fails(history, damage, error):
    """Every byte of an index line is held to the one form the writer produces."""
    store, sp, _ = history
    path = store.root / INDEX
    path.write_bytes(damage(path.read_bytes()))
    assert error in verify_store(store.root, sp).errors


def test_an_event_file_that_is_not_json_is_unreadable_not_missing(history):
    store, sp, _ = history
    (record,) = [r for r in store.iter_index(TRACK) if r["seq"] == 2]
    store.event_path(TRACK, record["event_id"]).write_bytes(b"\xff{")
    assert f"{INDEX}:2: event file unreadable (not UTF-8)" in verify_store(store.root, sp).errors


def test_a_manifest_that_hides_a_track_hides_nothing(history):
    """manifest.json is unsigned; the tracks verified are the spec's, whatever it lists."""
    store, sp, _ = history
    _flip_one_byte(store, 2, b"1", b"7")
    manifest = json.loads((store.root / "manifest.json").read_text())
    manifest["tracks"] = []
    (store.root / "manifest.json").write_text(json.dumps(manifest))
    (store.root / "tracks" / "elsewhere").mkdir()
    (store.root / "tracks" / "elsewhere" / "index-0000.jsonl").write_text("")

    report = verify_store(store.root, sp)
    assert not report.ok and report.records == 3
    assert f"{INDEX}:2: bad signature" in report.errors
    assert "manifest.json lists tracks [] but the spec's are ['franka_1arm']" in report.errors
    assert "tracks/elsewhere is not a track of the spec" in report.errors


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


def _event(store, seq):
    (record,) = [r for r in store.iter_index(TRACK) if r["seq"] == seq]
    path = store.event_path(TRACK, record["event_id"])
    return record, path, json.loads(path.read_text())


@pytest.mark.parametrize(
    "tamper",
    [
        lambda e: e.update(notes=["tampered"]),
        lambda e: e["units"][0].update(king_success=False, challenger_success=True),
        lambda e: e["units"][0]["prompt"].update(sha256="0" * 64),
        lambda e: e.update(finished_at="2026-09-14T00:00:00Z"),
    ],
)
def test_an_event_edited_after_its_record_was_signed_is_found(history, tamper):
    """The signed record carries the sha256 of its event's bytes, so nothing in an event - unit
    outcomes, prompt hashes, notes - can change unnoticed."""
    store, sp, _ = history
    record, path, event = _event(store, 2)
    assert record["event_sha256"]
    tamper(event)
    path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    assert f"{INDEX}:2: event file does not match the event_sha256 its record signs" in (
        verify_store(store.root, sp).errors
    )


def test_a_record_must_agree_with_the_event_it_signs(spec, tmp_path):
    """What the index duplicates from the event, the event must say too - every field of it,
    including the tally and media count its units add up to."""
    store = Store(tmp_path / "store", spec, Signer.generate())
    store.init(store.signer.verify_key_hex)
    record = make_record(spec, "duel", 1, KING, CHALLENGER)
    event = duel_event(
        record,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        units=[],
        units_per_skill=1,
        started_at="2026-09-13T11:00:00Z",
        wall_seconds=1.0,
    )
    store.write_event(TRACK, event)
    signed = {**record, "finished_at": "2026-09-14T00:00:00Z", "wins": 3, "decided": 3}
    store.append(TRACK, signed)
    errors = verify_store(store.root, spec).errors
    assert f"{INDEX}:1: event.finished_at differs from the index record" in errors
    assert f"{INDEX}:1: record wins 3, but the event's units tally 0" in errors
    assert f"{INDEX}:1: record decided 3, but the event's units tally 0" in errors


def test_a_record_that_predates_event_hashes_is_a_warning(history):
    store, sp, _ = history
    signer = store.signer
    path = store.root / INDEX
    lines = []
    for raw in path.read_text().splitlines():
        record = json.loads(raw.split("\t")[0])
        record.pop("event_sha256")
        body = canonical_json(record)
        lines.append(f"{body}\t{signer.sign(body)}\n")
    path.write_text("".join(lines))
    report = verify_store(store.root, sp)
    assert report.ok, report.errors
    assert f"{INDEX}:1: the record does not sign its event's bytes (no event_sha256)" in (
        report.warnings
    )


def test_a_clip_swapped_under_its_name_is_found(history):
    """Media paths are content addresses; the bytes are held to them."""
    store, sp, sha = history
    store.media_path(sha, "mp4").write_bytes(b"EVIL" * 10)
    assert verify_store(store.root, sp).errors == [
        f"{INDEX}:2: media {sha[:12]} content does not match its name"
    ]


@pytest.mark.parametrize(
    "field, value, error",
    [
        (
            "king",
            {"key": "e" * 16, "repo": "evil/king", "revision": "e" * 40},
            "head.json king differs from the king the signed records crown",
        ),
        ("block", 9, "head.json block differs from the last record"),
        (
            "finished_at",
            "2030-01-01T00:00:00Z",
            "head.json finished_at differs from the last record",
        ),
    ],
)
def test_the_head_is_held_to_what_the_signed_records_say(history, field, value, error):
    """head.json is unsigned and the dashboard renders it: every field of it is rebuilt from the
    signed index."""
    store, sp, _ = history
    path = store.head_path(TRACK)
    head = json.loads(path.read_text())
    head[field] = value
    path.write_text(json.dumps(head))
    assert f"tracks/{TRACK}/{error}" in verify_store(store.root, sp).errors


def test_an_empty_track_has_an_empty_head(spec, tmp_path):
    store = Store(tmp_path / "store", spec, Signer.generate())
    store.init(store.signer.verify_key_hex)
    store.write_head(TRACK, seq=0, event_id="", block=0, finished_at="", king=KING.as_dict())
    assert verify_store(store.root, spec).errors == [
        f"tracks/{TRACK}/head.json king differs from the king the signed records crown"
    ]


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
