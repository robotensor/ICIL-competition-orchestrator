"""Live frames, checked against the rules the dashboard's ingest applies.

`parse_live_frame` below is a line-by-line Python rendering of `parseLiveFrame` in
robotensor/robotensor-competition-dashboard, branch milestone-two-contests, `lib/live/types.ts`, and of the
body checks in `app/api/live/route.ts` (`MAX_BODY_BYTES`, JSON parse). The vocabularies it uses come
from `lib/protocol.ts` (`SIDES`, `OUTCOMES`) and `lib/protocol.generated.ts` (`LIVE_SCHEMA`, the
track ids and skill ids, generated from spec.json by `scripts/sync-spec.mjs`). If the dashboard
changes those rules, this file has to change with it.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vector_orchestrator.live import PHASES, LiveReporter, build_frame
from vector_orchestrator.spec import load_schema
from vector_orchestrator.store.verify import Report, SchemaCheck

# ---------------------------------------------------------------- the dashboard's rules

DASHBOARD_PHASES = ("fetching", "checking", "evaluating", "publishing", "done", "failed")
#: Phases the orchestrator reports that the dashboard does not accept yet: frames in them are
#: refused (422) until `lib/live/types.ts` adds them, and the duel goes on regardless.
AWAITING_DASHBOARD = ("materializing",)
DASHBOARD_SIDES = ("challenger", "king")
DASHBOARD_OUTCOMES = ("challenger", "king", "tie")
MAX_BODY_BYTES = 512 * 1024
# JavaScript's `$` matches the end of the string; Python's also matches before a final newline,
# so every one of these is used with `fullmatch`.
EVENT_ID = re.compile(r"[0-9a-f]{8,64}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def _is_record(v):
    return isinstance(v, dict)


def _coerce_skill(v, skills):
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    return s if s in skills else None


def _number(value):
    """`typeof value === 'number'`: a JSON number, and a bool is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _unit(v, skills):
    if not _is_record(v):
        return None
    skill = _coerce_skill(v.get("skill"), skills)
    task, instance = v.get("task"), v.get("instance")
    if not skill or not isinstance(task, str) or not task or not _number(instance):
        return None
    return {
        "skill": skill,
        "task": task,
        "demo_video": v["demo_video"] if SHA256.fullmatch(str(v.get("demo_video"))) else None,
        "outcome": v.get("outcome") if v.get("outcome") in DASHBOARD_OUTCOMES else None,
    }


def _current(v, skills):
    """`current()`: like a unit, but an empty task is allowed."""
    if not _is_record(v):
        return None
    skill = _coerce_skill(v.get("skill"), skills)
    if not skill or not isinstance(v.get("task"), str) or not _number(v.get("instance")):
        return None
    return {"skill": skill, "task": v["task"], "instance": v["instance"]}


def _recent_media(v, skills):
    if not _is_record(v) or not _is_record(v.get("unit")):
        return None
    unit = v["unit"]
    skill = _coerce_skill(unit.get("skill"), skills)
    if not skill or not isinstance(unit.get("task"), str) or not _number(unit.get("instance")):
        return None
    if v.get("side") not in DASHBOARD_SIDES:
        return None
    return {"unit": {"skill": skill, "task": unit["task"], "instance": unit["instance"]}}


def _model_ref(v):
    if not _is_record(v) or not isinstance(v.get("repo"), str) or not v["repo"]:
        return None
    return v


def parse_live_frame(raw, tracks, skills, live_schema):
    """`{ok: True, frame}` or `{ok: False, reason}`, as the dashboard decides."""
    if not _is_record(raw):
        return {"ok": False, "reason": "Body must be a JSON object."}
    if raw.get("schema") != live_schema or isinstance(raw.get("schema"), bool):
        return {"ok": False, "reason": f"schema must be {live_schema}."}
    key = raw.get("validator_key")
    if not isinstance(key, str) or not key.strip() or len(key) > 128:
        return {"ok": False, "reason": "validator_key must be a non-empty string."}
    if not isinstance(raw.get("track"), str) or raw["track"] not in tracks:
        return {"ok": False, "reason": f"track must be one of: {', '.join(tracks)}."}
    if not isinstance(raw.get("event_id"), str) or not EVENT_ID.fullmatch(raw["event_id"]):
        return {"ok": False, "reason": "event_id must be 8-64 lowercase hex characters."}
    if not isinstance(raw.get("phase"), str) or raw["phase"] not in DASHBOARD_PHASES:
        return {"ok": False, "reason": f"phase must be one of: {', '.join(DASHBOARD_PHASES)}."}
    side = raw.get("side")
    if side is not None and side not in DASHBOARD_SIDES:
        return {"ok": False, "reason": "side must be one of: challenger, king, or null."}
    units = [u for u in (_unit(i, skills) for i in raw.get("units") or []) if u]
    return {
        "ok": True,
        "frame": {
            **raw,
            "units": units,
            "current": _current(raw.get("current"), skills),
            "recent_media": _recent_media(raw.get("recent_media"), skills),
            "king": _model_ref(raw.get("king")),
            "challenger": _model_ref(raw.get("challenger")),
            "message": str(raw.get("message"))[:300],
        },
    }


def post_body_accepted(body: bytes, tracks, skills, live_schema):
    """`POST /api/live` after authentication: the size ceiling, the JSON parse, then the frame."""
    if len(body) > MAX_BODY_BYTES:
        return {"ok": False, "reason": "Frame too large."}
    try:
        raw = json.loads(body.decode("utf-8"))
    except ValueError:
        return {"ok": False, "reason": "Body is not valid JSON."}
    return parse_live_frame(raw, tracks, skills, live_schema)


# ---------------------------------------------------------------- frames under test

TRACK = "franka_1arm"
KING = {"key": "0123456789abcdef", "repo": "org/king", "revision": "a" * 40}
CHALLENGER = {"key": "fedcba9876543210", "repo": "org/challenger", "revision": "b" * 40}


def duel_units(spec):
    units = []
    for n, skill in enumerate(spec.skills(TRACK)):
        units.append(
            {
                "unit_id": f"{spec.skill_code(skill)}-{n:03d}",
                "skill": skill,
                "task": f"task_{n}",
                "task_label": f"Task {n}",
                "instance": 0,
                "king_success": True if n == 0 else None,
                "challenger_success": True if n == 0 else None,
                "outcome": "tie",
                "demo_video": "c" * 64,
                "king_video": None,
                "challenger_video": None,
                "void": n == 2,
            }
        )
    return units


def frame(spec, **overrides):
    kwargs = dict(
        track=TRACK,
        validator_key="d" * 64,
        event_id="e" * 64,
        kind="duel",
        duel_size="smoke",
        king=KING,
        challenger=CHALLENGER,
        phase="evaluating",
        side="king",
        units=duel_units(spec),
        current={
            "unit_id": "fs-001",
            "skill": "franka_stacking",
            "task": "task_1",
            "instance": 0,
            "demo_video": None,
        },
        recent_media={
            "unit": {"skill": "franka_pick_and_place", "task": "task_0", "instance": 0},
            "side": "king",
            "demo_video": "c" * 64,
            "video": "c" * 64,
            "success": True,
        },
        message="running franka_stacking",
        started_at="2026-09-13T11:00:00Z",
    )
    kwargs.update(overrides)
    return build_frame(spec, **kwargs)


def test_every_phase_and_side_builds_a_frame_the_dashboard_accepts_whole(spec):
    assert tuple(p for p in PHASES if p not in AWAITING_DASHBOARD) == DASHBOARD_PHASES
    reporter = LiveReporter(spec, "http://127.0.0.1:9", "tok")
    skills = spec.skills(TRACK)
    for phase, side in itertools.product(DASHBOARD_PHASES, (*DASHBOARD_SIDES, None)):
        built = frame(spec, phase=phase, side=side)
        parsed = post_body_accepted(reporter.encode(built), spec.tracks, skills, 4)
        assert parsed["ok"], (phase, side, parsed)
        kept = parsed["frame"]
        assert len(kept["units"]) == len(built["units"]), "the dashboard dropped units"
        for part in ("current", "recent_media", "king", "challenger"):
            assert kept[part] is not None, f"the dashboard dropped {part}"


def test_prompts_are_materialized_between_checking_and_evaluating(spec):
    """Both sides run from prompts fixed before either starts, and the live view says so. The
    dashboard has to learn the phase; until it does, it refuses these frames and nothing else."""
    assert PHASES.index("checking") < PHASES.index("materializing") < PHASES.index("evaluating")
    built = frame(spec, phase="materializing", side=None, current=None, recent_media=None)
    report = Report()
    SchemaCheck(load_schema()).check("LiveFrame", built, "frame", report)
    assert report.errors == []
    refused = parse_live_frame(built, spec.tracks, spec.skills(TRACK), 4)
    assert not refused["ok"] and "phase must be one of" in refused["reason"]


def test_the_frame_carries_progress_per_side_and_skill(spec):
    built = frame(spec, message="m" * 400)
    assert built["schema"] == spec.live["schema"] == 4
    assert built["total"] == 3
    assert built["skill_progress"]["king"]["franka_pick_and_place"] == {"done": 1, "total": 1}
    assert built["skill_progress"]["king"]["franka_press_push"] == {"done": 1, "total": 1}, "void"
    assert built["skill_progress"]["king"]["franka_stacking"] == {"done": 0, "total": 1}
    assert built["skill_progress"]["challenger"]["franka_stacking"] == {"done": 0, "total": 1}
    assert built["done"] == 2
    assert len(built["message"]) == 300
    # An outcome is only shown once both sides have run the unit.
    assert [u["outcome"] for u in built["units"]] == ["tie", None, None]
    report = Report()
    SchemaCheck(load_schema()).check("LiveFrame", built, "frame", report)
    assert report.errors == []


def test_a_large_unit_list_is_slimmed_to_fit_rather_than_refused(spec):
    reporter = LiveReporter(spec, "http://127.0.0.1:9", "tok")
    many = duel_units(spec) * 2000
    body = reporter.encode(frame(spec, units=many))
    assert len(body) <= min(spec.live["max_frame_bytes"], MAX_BODY_BYTES)
    assert post_body_accepted(body, spec.tracks, spec.skills(TRACK), 4)["ok"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"phase": "scoring"}, "phase must be one of"),
        ({"track": "sensorimotor"}, "track must be one of"),
        ({"event_id": "not-hex"}, "event_id must be 8-64 lowercase hex"),
        ({"event_id": "E" * 64}, "event_id must be 8-64 lowercase hex"),
        ({"side": "referee"}, "side must be one of"),
        ({"validator_key": ""}, "validator_key must be a non-empty string"),
        ({"units": [{"skill": "rt_stacking", "task": "t", "instance": 0}]}, "does not score"),
        # Python's `$` matches before a final newline; the dashboard's regex does not.
        ({"event_id": "deadbeef\n"}, "event_id must be 8-64 lowercase hex"),
        # Units the dashboard would silently drop, leaving a progress bar with rows missing.
        ({"units": [{"skill": "franka_stacking", "task": "t"}]}, "unit None: instance"),
        ({"units": [{"skill": "franka_stacking", "task": "", "instance": 0}]}, "task"),
        ({"units": [{"skill": "franka_stacking", "task": "t", "instance": True}]}, "instance"),
        ({"current": {"skill": "rt_stacking", "task": "t", "instance": 0}}, "current: skill"),
        ({"current": {"skill": "franka_stacking", "task": "t"}}, "current: instance"),
        (
            {"recent_media": {"unit": {"skill": "franka_stacking", "task": "t", "instance": 0}}},
            "recent_media: side",
        ),
        ({"king": {"key": "k", "repo": "", "revision": "r"}}, "king: repo"),
    ],
)
def test_a_frame_the_dashboard_would_refuse_or_thin_out_is_not_built(spec, overrides, message):
    with pytest.raises(ValueError, match=message):
        frame(spec, **overrides)


def test_the_encoded_rules_do_refuse_what_the_dashboard_refuses(spec):
    """The rules above are not vacuous: each strict field, broken, is refused."""
    good = frame(spec)
    skills = spec.skills(TRACK)
    for key, value in (
        ("schema", 3),
        ("validator_key", " "),
        ("track", "video_only"),
        ("event_id", "abc"),
        ("phase", "scoring"),
        ("side", "both"),
    ):
        assert not parse_live_frame({**good, key: value}, spec.tracks, skills, 4)["ok"], key
    unknown = {**good, "units": [{**good["units"][0], "skill": "rt_stacking"}]}
    assert parse_live_frame(unknown, spec.tracks, skills, 4)["frame"]["units"] == []
    for part, broken in (
        ("units", [{**good["units"][0], "task": ""}]),
        ("current", {**good["current"], "instance": None}),
        ("recent_media", {**good["recent_media"], "side": "referee"}),
        ("king", {**good["king"], "repo": ""}),
    ):
        thinned = parse_live_frame({**good, part: broken}, spec.tracks, skills, 4)["frame"][part]
        assert thinned in (None, []), part


class _Sink(BaseHTTPRequestHandler):
    frames: list = []

    def log_message(self, *a):
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Sink.frames.append(
            (self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(n)))
        )
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_GET(self):
        _Sink.frames.append((self.path, self.headers.get("Authorization"), {}))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


class _Slow(BaseHTTPRequestHandler):
    """Answers a byte at a time, for longer than any duel would wait."""

    def log_message(self, *a):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Length", "20")
        self.end_headers()
        for _ in range(20):
            try:
                self.wfile.write(b"x")
                self.wfile.flush()
            except OSError:
                return
            time.sleep(0.5)


class _Redirect(BaseHTTPRequestHandler):
    elsewhere = ""

    def log_message(self, *a):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(302)
        self.send_header("Location", _Redirect.elsewhere)
        self.send_header("Content-Length", "0")
        self.end_headers()


@contextmanager
def serving(handler):
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def test_a_slow_dashboard_does_not_hold_the_duel(spec):
    """The timeout is the whole request's, not each socket read's: a window that never finishes
    answering must not stop a duel."""
    with serving(_Slow) as url:
        rep = LiveReporter(spec, url, "tok", timeout_s=1.0)
        started = time.monotonic()
        assert not rep.post(frame(spec), force=True)
        assert time.monotonic() - started < 3, "the reporter waited on a slow answer"
        assert rep.failed == 1


def test_the_bearer_token_never_follows_a_redirect(spec):
    """urllib would replay the POST as a GET and carry the Authorization header to whatever host
    the answer names; the live token opens the dashboard's ingest."""
    _Sink.frames = []
    with serving(_Sink) as elsewhere, serving(_Redirect) as url:
        _Redirect.elsewhere = f"{elsewhere}/steal"
        rep = LiveReporter(spec, url, "SECRET-TOKEN", timeout_s=2.0)
        assert not rep.post(frame(spec), force=True) and rep.failed == 1
    assert _Sink.frames == [], "the token was sent to another host"


def test_the_reporter_posts_with_a_bearer_token_and_never_raises(spec):
    built = frame(spec)
    _Sink.frames = []
    httpd = HTTPServer(("127.0.0.1", 0), _Sink)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        rep = LiveReporter(spec, f"http://127.0.0.1:{httpd.server_address[1]}", "tok")
        assert rep.enabled and rep.post(built, force=True)
        assert not rep.post(built)  # rate limited
        assert rep.post(built, force=True)
        path, auth, got = _Sink.frames[0]
        assert path == spec.live["path"] and auth == "Bearer tok" and got["event_id"] == "e" * 64
        off = LiveReporter(spec, None, None)
        assert not off.enabled and not off.post(built, force=True)
        dead = LiveReporter(spec, "http://127.0.0.1:9", "tok", timeout_s=0.5)
        assert not dead.post(built, force=True) and dead.failed == 1
    finally:
        httpd.shutdown()
