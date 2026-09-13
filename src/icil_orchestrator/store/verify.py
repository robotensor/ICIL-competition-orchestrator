"""Independent verification of a store: signatures, sequence, events, media, schema.

What a third party runs against a published store. Every problem is reported with the file, and
for an index line its line number, so a tampered or corrupted record is found, not just detected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..canon import canonical_json, sha256_file, verify_signature
from ..spec import Spec, load_schema
from .records import media_shas
from .writer import Store


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    records: int = 0
    events: int = 0
    media: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


#: The one form a signature takes in an index line: an ed25519 signature is 64 bytes.
SIGNATURE_RE = re.compile(r"[0-9a-f]{128}")


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON a signed record can hold")


def parse_json_bytes(data: bytes) -> tuple[Any, str | None]:
    """`(document, None)`, or `(None, why)` for bytes that are not strict UTF-8 JSON - never an
    exception, whatever the bytes: a verifier names damage, it does not crash on it."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, "not UTF-8"
    try:
        return json.loads(text, parse_constant=_refuse_constant), None
    except (ValueError, RecursionError):
        return None, "unparsable"


def read_json_file(path: Path) -> tuple[Any, str | None]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"unreadable ({exc.strerror or exc})"
    return parse_json_bytes(data)


class SchemaCheck:
    """Validates a document against one `$defs` entry of `store-schema.json`."""

    def __init__(self, schema: dict[str, Any]):
        import jsonschema

        self._jsonschema = jsonschema
        self._defs = schema["$defs"]
        self._validators: dict[str, Any] = {}

    def check(self, ref: str, obj: Any, where: str, report: Report) -> None:
        validator = self._validators.get(ref)
        if validator is None:
            sub = {"$ref": f"#/$defs/{ref}", "$defs": self._defs}
            validator = self._validators[ref] = self._jsonschema.Draft202012Validator(sub)
        for err in validator.iter_errors(obj):
            path = "/".join(str(p) for p in err.path)
            report.errors.append(f"{where}: schema {ref}: {err.message} at {path}")


def verify_store(root: str | Path, spec: Spec, schema: dict[str, Any] | None = None) -> Report:
    report = Report()
    store = Store(root, spec)
    validator = SchemaCheck(schema if schema is not None else load_schema())

    manifest, _ = read_json_file(store.root / "manifest.json")
    if not isinstance(manifest, dict):
        report.errors.append("manifest.json missing or unreadable")
        return report
    validator.check("Manifest", manifest, "manifest.json", report)
    key = str(manifest.get("validator_key", ""))
    if manifest.get("spec_fingerprint") != spec.fingerprint:
        report.warnings.append("manifest spec_fingerprint differs from the loaded spec.json")

    video_ext = spec.video_format
    #: Each clip is hashed once, however many units and events refer to it.
    hashed: dict[str, bool] = {}
    # manifest.json is unsigned, so it cannot choose what is verified: the tracks are the spec's,
    # and anything the store holds for another track is reported rather than skipped.
    listed = manifest.get("tracks")
    if listed != list(spec.tracks):
        report.errors.append(
            f"manifest.json lists tracks {listed!r} but the spec's are {list(spec.tracks)!r}"
        )
    for top in ("tracks", "events"):
        folder = store.root / top
        for entry in sorted(folder.iterdir()) if folder.is_dir() else ():
            if entry.name not in spec.tracks:
                report.errors.append(f"{top}/{entry.name} is not a track of the spec")
    for track in spec.tracks:
        expected_seq = 1
        last: dict[str, Any] | None = None
        part = 0
        while True:
            path = store.index_part_path(track, part)
            if not path.exists():
                break
            # Bytes, split on the newline the writer ends every line with, and each line decoded
            # on its own: one bad byte is that line's error, never the whole part's exception.
            lines = path.read_bytes().split(b"\n")
            for n, raw_bytes in enumerate(lines, start=1):
                where = f"{path.relative_to(store.root)}:{n}"
                if n == len(lines):
                    if not raw_bytes:
                        break
                    report.errors.append(f"{where}: no newline at the end of the line")
                if not raw_bytes:
                    report.errors.append(f"{where}: blank line")
                    continue
                try:
                    raw = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    report.errors.append(f"{where}: not UTF-8")
                    continue
                if "\t" not in raw:
                    report.errors.append(f"{where}: no signature")
                    continue
                body, sig = raw.split("\t", 1)
                record, _ = parse_json_bytes(body.encode("utf-8"))
                if record is None:
                    report.errors.append(f"{where}: unparsable record")
                    continue
                if not isinstance(record, dict):
                    report.errors.append(f"{where}: record is not an object")
                    continue
                try:
                    canonical = canonical_json(record) == body
                except (ValueError, RecursionError):
                    canonical = False
                if not canonical:
                    report.errors.append(f"{where}: record is not canonical JSON")
                # The signature is held to its one encoding as strictly as the body is: upper-case
                # hex or trailing whitespace verify the same bytes, but they are not the line the
                # writer signed.
                if not SIGNATURE_RE.fullmatch(sig):
                    report.errors.append(f"{where}: signature is not 128 lowercase hex characters")
                elif not verify_signature(key, body, sig):
                    report.errors.append(f"{where}: bad signature")
                if record.get("seq") != expected_seq:
                    report.errors.append(
                        f"{where}: seq {record.get('seq')} expected {expected_seq}"
                    )
                seq = record.get("seq")
                expected_seq = (seq if isinstance(seq, int) else expected_seq) + 1
                if isinstance(seq, int) and store.part_of(seq) != part:
                    report.errors.append(f"{where}: record filed in the wrong part")
                if record.get("track") != track:
                    report.errors.append(
                        f"{where}: track {record.get('track')!r} filed under {track}"
                    )
                validator.check("IndexRecord", record, where, report)
                report.records += 1
                last = record

                event_where = f"events/{track}/{record.get('event_id')}.json"
                event, problem = read_json_file(
                    store.event_path(track, str(record.get("event_id", "")))
                )
                if problem == "missing":
                    report.errors.append(f"{where}: event file missing")
                    continue
                if not isinstance(event, dict):
                    report.errors.append(
                        f"{where}: event file unreadable ({problem or 'not an object'})"
                    )
                    continue
                report.events += 1
                validator.check("DuelEvent", event, event_where, report)
                for k in ("event_id", "kind", "block", "dethroned", "king", "challenger"):
                    if event.get(k) != record.get(k):
                        report.errors.append(f"{where}: event.{k} differs from the index record")
                for sha in media_shas(event.get("units", [])):
                    if not store.has_media(sha, video_ext):
                        report.errors.append(f"{where}: media {sha[:12]} missing")
                        continue
                    report.media += 1
                    if sha not in hashed:
                        hashed[sha] = sha256_file(store.media_path(sha, video_ext)) == sha
                    if not hashed[sha]:
                        report.errors.append(
                            f"{where}: media {sha[:12]} content does not match its name"
                        )
            part += 1

        head, _ = read_json_file(store.head_path(track))
        if not isinstance(head, dict):
            report.errors.append(f"tracks/{track}/head.json missing or unreadable")
        else:
            validator.check("Head", head, f"tracks/{track}/head.json", report)
            if last is not None and (
                head.get("seq") != last.get("seq") or head.get("event_id") != last.get("event_id")
            ):
                report.errors.append(f"tracks/{track}/head.json does not point at the last record")
            if last is None and head.get("seq") not in (0, None):
                report.errors.append(f"tracks/{track}/head.json claims records that do not exist")
        queue = store.queue_path(track)
        if queue.exists():
            snapshot, problem = read_json_file(queue)
            if problem is not None:
                report.warnings.append(
                    f"tracks/{track}/queue.json unreadable (rewritten every cycle; may be mid-write)"
                )
            else:
                validator.check("QueueSnapshot", snapshot, f"tracks/{track}/queue.json", report)
    return report
