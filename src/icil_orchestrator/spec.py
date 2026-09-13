"""The contract: `spec.json`, read once, validated, never duplicated as literals.

`spec.json` and `store-schema.json` are shared with the dashboard, which renders what they say. So
every number the orchestrator acts on is read through `Spec`, and `validate_spec` refuses a contract
the orchestrator could not honour rather than letting it half-work.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .benchmarks.api import BENCHMARK_API_VERSION
from .canon import canonical_sha256

SPEC_ENV = "ICIL_ORCHESTRATOR_SPEC"
SCHEMA_ENV = "ICIL_ORCHESTRATOR_STORE_SCHEMA"
SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SKILL_CODE_RE = re.compile(r"^[a-z]{2}$")
TRACK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TRACK_CODE_RE = re.compile(r"^[a-z]{2}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
BENCHMARK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

#: How a track stops a policy from simply replaying the demonstration it was shown.
PROTOCOLS = ("different_initial_state", "same_initial_state")

#: What a track's policies may see of a demonstration.
DEMO_VIEWS = ("sensorimotor", "video_only")

#: Where a track's prompts come from. Only "materialized" exists here: prompts are produced once
#: per duel by the benchmark and published with the event. Pools were the weights-era alternative.
PROMPT_SOURCES = ("materialized",)

#: Budgets the orchestrator enforces, each a positive number of seconds.
BUDGETS = (
    "policy_start_seconds",
    "act_timeout_s",
    "materialize_wall_seconds",
    "unit_wall_seconds",
    "side_wall_seconds",
    "duel_wall_seconds",
)


def _non_root(value: Any) -> bool:
    """A container user that is neither uid 0 nor gid 0, however it is spelled.

    Docker takes `user[:group]`, numeric or by name, and `00` is uid 0 as surely as `0` is; a
    submission running as root inside the sandbox is the one thing the sandbox is for.
    """
    if not isinstance(value, str) or not value:
        return False
    parts = value.split(":")
    if len(parts) > 2:
        return False
    for part in parts:
        name = part.strip()
        if not name or name == "root":
            return False
        if name.isdigit() and int(name) == 0:
            return False
    return True


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "spec.json").exists():
            return parent
    return None


def _resolve(name: str, env: str) -> Path:
    explicit = os.environ.get(env)
    if explicit:
        return Path(explicit)
    packaged = Path(__file__).resolve().parent / "data" / name
    if packaged.exists():
        return packaged
    root = _repo_root()
    if root and (root / name).exists():
        return root / name
    raise FileNotFoundError(f"{name} not found; set {env}")


def spec_path() -> Path:
    return _resolve("spec.json", SPEC_ENV)


def schema_path() -> Path:
    return _resolve("store-schema.json", SCHEMA_ENV)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def validate_spec(doc: dict[str, Any]) -> list[str]:
    """Every problem with a contract, as short paths. Empty when the orchestrator can honour it.

    A benchmark that is not installed is accepted on purpose: CI, a laptop and the dashboard's
    vendored copy must all be able to check the contract with no benchmark present. Whether the
    installed one matches its pin is `icil-orchestrator benchmarks check`.
    """
    errors: list[str] = []

    def need(path: str, cond: bool) -> None:
        if not cond:
            errors.append(path)

    need("spec_version:int", _positive_int(doc.get("spec_version")))
    need("track removed (v3); use tracks", "track" not in doc)
    for gone in ("model", "pools"):
        need(f"{gone} removed (v7); submissions are code, see submission", gone not in doc)

    benchmarks = doc.get("benchmarks") or {}
    need("benchmarks non-empty", isinstance(benchmarks, dict) and bool(benchmarks))
    for name, entry in benchmarks.items() if isinstance(benchmarks, dict) else ():
        if name.startswith("_"):
            continue
        need(f"benchmarks.{name} id", bool(BENCHMARK_ID_RE.match(name)))
        if not isinstance(entry, dict):
            errors.append(f"benchmarks.{name}: mapping")
            continue
        need(f"benchmarks.{name}.distribution", _text(entry.get("distribution")))
        need(
            f"benchmarks.{name}.api_version == {BENCHMARK_API_VERSION}",
            entry.get("api_version") == BENCHMARK_API_VERSION,
        )
        version = entry.get("version")
        need(f"benchmarks.{name}.version str|null", version is None or _text(version))
        wheel = entry.get("wheel_sha256")
        need(
            f"benchmarks.{name}.wheel_sha256 hex64|null",
            wheel is None or (isinstance(wheel, str) and bool(HEX64_RE.match(wheel))),
        )

    skills = doc.get("skills") or {}
    need("skills non-empty", isinstance(skills, dict) and bool(skills))
    codes: set[str] = set()
    for sid, s in skills.items() if isinstance(skills, dict) else ():
        need(f"skills.{sid} id", bool(SKILL_ID_RE.match(sid)))
        if not isinstance(s, dict):
            errors.append(f"skills.{sid}: mapping")
            continue
        code = s.get("code")
        need(f"skills.{sid}.code", isinstance(code, str) and bool(SKILL_CODE_RE.match(code)))
        need(f"skills.{sid}.code unique", code not in codes)
        codes.add(str(code))
        for key in ("title", "blurb", "suite"):
            need(f"skills.{sid}.{key}", _text(s.get(key)))
        need(f"skills.{sid}.category", s.get("category") is None or _text(s.get("category")))
        bench = s.get("benchmark")
        need(f"skills.{sid}.benchmark declared", isinstance(bench, str) and bench in benchmarks)
        need(f"skills.{sid}.max_steps", _positive_int(s.get("max_steps")))
        need(f"skills.{sid}.environment", isinstance(s.get("environment"), dict))
        for gone in ("architecture", "simulator", "tasks", "perturbations"):
            need(f"skills.{sid}.{gone} removed", gone not in s)

    duel = doc.get("duel") or {}
    sizes = duel.get("sizes") or {}
    need("duel.default_size in sizes", duel.get("default_size") in sizes)
    for name, entry in sizes.items():
        if not isinstance(entry, dict):
            need(f"duel.sizes.{name} is an object", False)
            continue
        n = entry.get("units_per_skill")
        need(f"duel.sizes.{name}.units_per_skill>=1", _positive_int(n))
    margin = duel.get("score_margin")
    need("duel.score_margin in [0,100]", isinstance(margin, (int, float)) and 0 <= margin <= 100)
    void = duel.get("max_void_fraction")
    need("duel.max_void_fraction in [0,1]", isinstance(void, (int, float)) and 0 <= void <= 1)

    tracks = doc.get("tracks") or {}
    need("tracks non-empty", isinstance(tracks, dict) and bool(tracks))
    claimed: list[str] = []
    slugs: set[str] = set()
    track_codes: set[str] = set()
    for tid, t in tracks.items() if isinstance(tracks, dict) else ():
        need(f"tracks.{tid} id", bool(TRACK_ID_RE.match(tid)))
        if not isinstance(t, dict):
            errors.append(f"tracks.{tid}: mapping")
            continue
        need(f"tracks.{tid}.id == key", t.get("id") == tid)
        need(
            f"tracks.{tid}.code",
            isinstance(t.get("code"), str) and bool(TRACK_CODE_RE.match(t["code"])),
        )
        need(f"tracks.{tid}.code unique", t.get("code") not in track_codes)
        track_codes.add(str(t.get("code")))
        need(f"tracks.{tid}.slug", bool(SLUG_RE.match(str(t.get("slug", "")))))
        need(f"tracks.{tid}.slug unique", t.get("slug") not in slugs)
        slugs.add(str(t.get("slug")))
        for key in ("short", "title", "blurb"):
            need(f"tracks.{tid}.{key}", _text(t.get(key)))
        need(f"tracks.{tid}.k_demos==1", t.get("k_demos") == 1)
        need(f"tracks.{tid}.language==none", t.get("language") == "none")
        need(f"tracks.{tid}.protocol", t.get("protocol") in PROTOCOLS)
        need(f"tracks.{tid}.prompts", t.get("prompts") in PROMPT_SOURCES)
        need(
            f"tracks.{tid}.prompt_instance_disjoint",
            isinstance(t.get("prompt_instance_disjoint"), bool),
        )
        # Same Scene shows the demonstration of the very state it scores; only a track that does
        # not may claim the two are disjoint. Getting this pair backwards misdescribes every score.
        need(
            f"tracks.{tid}.prompt_instance_disjoint matches protocol",
            t.get("prompt_instance_disjoint") == (t.get("protocol") == "different_initial_state"),
        )
        demo = t.get("demonstration") or {}
        need(f"tracks.{tid}.demonstration.view", demo.get("view") in DEMO_VIEWS)
        need(
            f"tracks.{tid}.demonstration.modalities has video",
            isinstance(demo.get("modalities"), list) and "video" in demo["modalities"],
        )
        need(f"tracks.{tid}.demonstration.withheld", isinstance(demo.get("withheld"), list))
        track_skills = t.get("skills")
        need(
            f"tracks.{tid}.skills non-empty", isinstance(track_skills, list) and bool(track_skills)
        )
        for sid in track_skills or []:
            need(f"tracks.{tid}.skills.{sid} exists", sid in skills)
            claimed.append(sid)
        own_sizes = t.get("sizes")
        if own_sizes is not None:
            need(
                f"tracks.{tid}.sizes names match duel.sizes",
                isinstance(own_sizes, dict) and set(own_sizes) == set(sizes),
            )
            for name, entry in (own_sizes or {}).items():
                if not isinstance(entry, dict):
                    need(f"tracks.{tid}.sizes.{name} is an object", False)
                    continue
                need(
                    f"tracks.{tid}.sizes.{name}.units_per_skill>=1",
                    _positive_int(entry.get("units_per_skill")),
                )
        need(
            f"tracks.{tid}.default_size in sizes",
            t.get("default_size") in (own_sizes if own_sizes is not None else sizes),
        )
        for key, lo, hi in (("score_margin", 0, 100), ("max_void_fraction", 0, 1)):
            if key in t:
                value = t[key]
                need(
                    f"tracks.{tid}.{key} in [{lo},{hi}]",
                    isinstance(value, (int, float)) and lo <= value <= hi,
                )
    # Every skill belongs to exactly one track: one scored nowhere would sit in the contract
    # affecting nothing, and the dashboard refuses a spec where that is not so.
    need("tracks partition the skills", sorted(set(claimed)) == sorted(skills))
    need("tracks claim no skill twice", len(claimed) == len(set(claimed)))

    baselines = doc.get("baselines")
    need("baselines", isinstance(baselines, dict))
    for tid in tracks if isinstance(tracks, dict) else ():
        need(f"baselines.{tid}", isinstance(baselines, dict) and tid in baselines)
        entry = (baselines or {}).get(tid) if isinstance(baselines, dict) else None
        need(
            f"baselines.{tid} null or {{repo, revision}}",
            entry is None or (isinstance(entry, dict) and _text(entry.get("repo"))),
        )

    budgets = doc.get("budgets") or {}
    for key in BUDGETS:
        need(f"budgets.{key}>0", _positive_number(budgets.get(key)))

    submission = doc.get("submission") or {}
    need("submission.manifest", _text(submission.get("manifest")))
    need("submission.manifest_api:int", _positive_int(submission.get("manifest_api")))
    need("submission.policy_protocol:int", _positive_int(submission.get("policy_protocol")))
    need("submission.max_repo_bytes", _positive_int(submission.get("max_repo_bytes")))
    image = submission.get("base_image") or {}
    need("submission.base_image.name", _text(image.get("name")))
    digest = image.get("digest")
    need(
        "submission.base_image.digest sha256:<hex>|null",
        digest is None or (isinstance(digest, str) and bool(IMAGE_DIGEST_RE.match(digest))),
    )
    sandbox = submission.get("sandbox") or {}
    # The sandbox is the whole of what an untrusted submission is held to; a contract that loosens
    # it is refused here rather than trusted to be read correctly by the container runner.
    need("submission.sandbox.network == none", sandbox.get("network") == "none")
    need("submission.sandbox.read_only_root", sandbox.get("read_only_root") is True)
    need("submission.sandbox.tmpfs", isinstance(sandbox.get("tmpfs"), list))
    need("submission.sandbox.user non-root", _non_root(sandbox.get("user")))
    for key in ("gpus", "memory_bytes", "cpus", "pids"):
        need(f"submission.sandbox.{key}>0", _positive_number(sandbox.get(key)))

    media = doc.get("media") or {}
    need("media.video.format", _text((media.get("video") or {}).get("format")))

    store = doc.get("store") or {}
    need("store.schema:int", _positive_int(store.get("schema")))
    need("store.index_lines_per_part>=1", _positive_int(store.get("index_lines_per_part")))
    need("store.media_bucket_hex in 1..4", store.get("media_bucket_hex") in (1, 2, 3, 4))

    live = doc.get("live") or {}
    need("live.schema:int", _positive_int(live.get("schema")))
    need("live.path", isinstance(live.get("path"), str) and live["path"].startswith("/"))
    need("live.max_frame_bytes", _positive_int(live.get("max_frame_bytes")))
    return errors


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    path: Path
    fingerprint: str

    # -- tracks
    @property
    def version(self) -> int:
        return int(self.raw["spec_version"])

    @property
    def tracks(self) -> tuple[str, ...]:
        """The tracks of the competition, in declaration order."""
        return tuple(self.raw["tracks"].keys())

    def track(self, track: str) -> dict[str, Any]:
        try:
            return self.raw["tracks"][track]
        except KeyError:
            raise KeyError(
                f"unknown track {track!r}; the tracks are {', '.join(self.tracks)}"
            ) from None

    @property
    def sole_track(self) -> str:
        """The only track, for a command line that was not told which one.

        Raises rather than guessing the moment a second track is declared, so an entry cannot
        quietly be queued for, or scored in, the wrong one.
        """
        tracks = self.tracks
        if len(tracks) != 1:
            raise ValueError(
                f"the spec declares {len(tracks)} tracks ({', '.join(tracks)}); say which one"
            )
        return tracks[0]

    def track_title(self, track: str) -> str:
        return str(self.track(track)["title"])

    def track_of(self, skill: str) -> str:
        """The track a skill is scored in. Total for any skill in the contract."""
        for tid in self.tracks:
            if skill in self.track(tid)["skills"]:
                return tid
        raise KeyError(f"skill {skill!r} belongs to no track")

    def demonstration(self, track: str) -> dict[str, Any]:
        """What this track's policies are shown of a demonstration: view, modalities, withheld."""
        return dict(self.track(track)["demonstration"])

    def demo_view(self, track: str) -> str:
        return str(self.track(track)["demonstration"]["view"])

    def protocol(self, track: str) -> str:
        return str(self.track(track)["protocol"])

    def prompts(self, track: str) -> str:
        return str(self.track(track)["prompts"])

    def benchmarks_of(self, track: str) -> tuple[str, ...]:
        """The benchmarks this track's skills run on, in first-use order."""
        return tuple(dict.fromkeys(self.benchmark_of(s) for s in self.skills(track)))

    # -- skills
    @property
    def all_skills(self) -> tuple[str, ...]:
        """Every skill in the contract, in declaration order."""
        return tuple(self.raw["skills"].keys())

    def skills(self, track: str) -> tuple[str, ...]:
        """The skills one track scores, in the order they are run, rendered and averaged."""
        return tuple(self.track(track)["skills"])

    def skill(self, name: str) -> dict[str, Any]:
        return self.raw["skills"][name]

    def skill_code(self, name: str) -> str:
        return str(self.skill(name)["code"])

    def skill_for_code(self, code: str) -> str:
        for s in self.all_skills:
            if self.skill_code(s) == code:
                return s
        raise KeyError(code)

    def skill_title(self, name: str) -> str:
        return str(self.skill(name)["title"])

    def benchmark_of(self, name: str) -> str:
        """The benchmark (entry point name) a skill's units come from."""
        return str(self.skill(name)["benchmark"])

    def suite(self, name: str) -> str:
        return str(self.skill(name)["suite"])

    def category(self, name: str) -> str | None:
        category = self.skill(name).get("category")
        return None if category is None else str(category)

    def max_steps(self, name: str) -> int:
        return int(self.skill(name)["max_steps"])

    def env(self, name: str) -> dict[str, Any]:
        return self.skill(name)["environment"]

    # -- benchmarks
    @property
    def benchmarks(self) -> dict[str, dict[str, Any]]:
        """`benchmarks`: the pin each benchmark's distribution must match."""
        return {
            k: dict(v)
            for k, v in self.raw["benchmarks"].items()
            if not k.startswith("_") and isinstance(v, dict)
        }

    def benchmark_pin(self, name: str) -> dict[str, Any] | None:
        return self.benchmarks.get(name)

    # -- duel
    @property
    def duel(self) -> dict[str, Any]:
        return self.raw["duel"]

    def _duel_of(self, track: str, key: str) -> Any:
        """A track's own value for a duelling constant, or the competition-wide default."""
        own = self.track(track).get(key)
        return self.raw["duel"][key] if own is None else own

    def sizes(self, track: str) -> tuple[str, ...]:
        return tuple(self._duel_of(track, "sizes").keys())

    def default_size(self, track: str) -> str:
        return str(self.track(track)["default_size"])

    def size_of(self, track: str, size: str | None) -> str:
        return size if size in self._duel_of(track, "sizes") else self.default_size(track)

    def units_per_skill(self, track: str, size: str | None = None) -> int:
        sizes = self._duel_of(track, "sizes")
        return int(sizes[self.size_of(track, size)]["units_per_skill"])

    def units_per_side(self, track: str, size: str | None = None) -> int:
        return self.units_per_skill(track, size) * len(self.skills(track))

    def score_margin(self, track: str) -> float:
        return float(self._duel_of(track, "score_margin"))

    def max_void_fraction(self, track: str) -> float:
        return float(self._duel_of(track, "max_void_fraction"))

    # -- the rest
    @property
    def budgets(self) -> dict[str, Any]:
        return self.raw["budgets"]

    @property
    def submission(self) -> dict[str, Any]:
        return self.raw["submission"]

    @property
    def media(self) -> dict[str, Any]:
        return self.raw["media"]

    @property
    def video_format(self) -> str:
        return str(self.raw["media"]["video"]["format"])

    @property
    def store(self) -> dict[str, Any]:
        return self.raw["store"]

    @property
    def live(self) -> dict[str, Any]:
        return self.raw["live"]

    def baseline(self, track: str) -> dict[str, Any] | None:
        """This track's genesis king, or None where the throne opens empty."""
        entry = (self.raw.get("baselines") or {}).get(track)
        return dict(entry) if isinstance(entry, dict) else None


class SpecError(ValueError):
    """A contract the orchestrator cannot honour."""


def load_spec_file(path: str | Path) -> Spec:
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    errors = validate_spec(doc)
    if errors:
        raise SpecError(f"{p}: invalid spec: " + ", ".join(errors))
    return Spec(raw=doc, path=p, fingerprint=canonical_sha256(doc))


@lru_cache(maxsize=4)
def _cached(path: str) -> Spec:
    return load_spec_file(path)


def load_spec(path: str | Path | None = None) -> Spec:
    return _cached(str(Path(path) if path else spec_path()))


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or schema_path()).read_text(encoding="utf-8"))
