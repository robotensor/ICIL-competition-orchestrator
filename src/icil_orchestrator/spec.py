"""The contract: `spec.json`, read once, validated, never duplicated as literals.

`spec.json` and `store-schema.json` are shared with the dashboard, which renders what they say. So
every number the orchestrator acts on is read through `Spec`, and `validate_spec` refuses a contract
the orchestrator could not honour rather than letting it half-work.
"""

from __future__ import annotations

import json
import os
import posixpath
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

#: What a policy may answer `hello` with, and what `environment.action_dims` is keyed by. The wire
#: carries one action array per step or chunk, of the dimension its type names.
ACTION_TYPES = ("qpos", "ee")

#: How a track stops a policy from simply replaying the demonstration it was shown.
PROTOCOLS = ("different_initial_state", "same_initial_state")

#: What a track's policies may see of a demonstration. Only the whole demonstration: withholding a
#: channel (icilval's `video_only`) needs the demonstration view that was not ported, so a spec
#: asking for one would publish a view nothing enforces.
DEMO_VIEWS = ("sensorimotor",)

#: Where a track's prompts come from. Only "materialized" exists here: prompts are produced once
#: per duel by the benchmark and published with the event. Pools were the weights-era alternative.
PROMPT_SOURCES = ("materialized",)

#: What a submission is: `code`, a repository with `icil.yaml`, its policy code and weights, run
#: in the Docker sandbox; or `weights`, a repository holding only the weights of an architecture
#: the validator owns the code of (`submission.model`), served by that code with no sandbox,
#: since nothing of the submission's ever runs.
SUBMISSION_KINDS = ("code", "weights")
#: What a crown rule may hold paired outcomes to (`duel.crown.paired_test`).
PAIRED_TESTS = ("sign",)
MODULE_CLASS_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")

#: Budgets the orchestrator enforces, each a positive number of seconds.
BUDGETS = (
    "policy_start_seconds",
    "act_timeout_s",
    "materialize_wall_seconds",
    "unit_wall_seconds",
    "policy_budget_seconds",
    "side_wall_seconds",
    "duel_wall_seconds",
)


def _environment_errors(where: str, env: dict[str, Any], need) -> None:
    """A skill's environment is the demonstration's shape: what the prompt holds and what a policy
    must answer with. The benchmark builds scenes from `embodiment`, the dashboard lays clips out
    by `cameras`, and a policy's actions are checked against `action_dims`, so a typo here is a
    duel that voids rather than a contract that fails to load."""
    embodiment = env.get("embodiment")
    need(
        f"{where}.embodiment [left arm, right arm, distance|null]",
        isinstance(embodiment, list)
        and len(embodiment) == 3
        and all(_text(arm) for arm in embodiment[:2])
        and (embodiment[2] is None or _number(embodiment[2])),
    )
    cameras = env.get("cameras")
    need(
        f"{where}.cameras non-empty list of names",
        isinstance(cameras, list) and bool(cameras) and all(_text(c) for c in cameras),
    )
    types = env.get("action_types")
    need(
        f"{where}.action_types non-empty subset of {list(ACTION_TYPES)}",
        isinstance(types, list) and bool(types) and all(t in ACTION_TYPES for t in types),
    )
    dims = env.get("action_dims")
    need(f"{where}.action_dims mapping", isinstance(dims, dict))
    for action_type in types if isinstance(types, list) else ():
        need(
            f"{where}.action_dims.{action_type}>0",
            isinstance(dims, dict) and _positive_int(dims.get(action_type)),
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


#: Where a `sandbox.tmpfs` mount may be neither at nor under: the kernel's and the runtime's trees
#: (runc refuses a tmpfs over /proc, and one over /sys or /dev hides what the runtime put there),
#: the checkout (`submissions.image.SUBMISSION_DIR`), which a tmpfs would empty, and the socket's
#: directory (`submissions.container.SOCKET_DIR`), which one would hide from the host. Docker
#: refuses `/` itself.
SANDBOX_RESERVED_PATHS = ("/proc", "/sys", "/dev", "/submission", "/run/icil")


def _tmpfs_path(value: Any) -> bool:
    """A path `docker run --tmpfs PATH:OPTIONS` takes whole and as spelled: absolute, normalised
    (no trailing or doubled slash, no `.` or `..`), and free of the colon that starts its options,
    the comma between them and whitespace."""
    return (
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and posixpath.normpath(value) == value
        and not any(c in value for c in ":,")
        and not any(c.isspace() for c in value)
    )


def _tmpfs_reserved(path: str) -> bool:
    """Whether a tmpfs at `path` would be `/` or at or under one of `SANDBOX_RESERVED_PATHS`."""
    return path == "/" or any(
        path == reserved or path.startswith(reserved + "/") for reserved in SANDBOX_RESERVED_PATHS
    )


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


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _code_submission_errors(submission: dict[str, Any], need) -> None:
    """A code submission's manifest, base image and sandbox (`SUBMISSION_KINDS`)."""
    need("submission.manifest", _text(submission.get("manifest")))
    need("submission.manifest_api:int", _positive_int(submission.get("manifest_api")))
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
    tmpfs = sandbox.get("tmpfs")
    paths = (
        tmpfs if isinstance(tmpfs, list) and tmpfs and all(_tmpfs_path(p) for p in tmpfs) else None
    )
    need("submission.sandbox.tmpfs non-empty list of absolute paths", paths is not None)
    if paths is not None:
        need("submission.sandbox.tmpfs paths distinct", len(set(paths)) == len(paths))
        need(
            "submission.sandbox.tmpfs not / and not at or under "
            + ", ".join(SANDBOX_RESERVED_PATHS),
            not any(_tmpfs_reserved(p) for p in paths),
        )
    need("submission.sandbox.tmpfs_exec bool", isinstance(sandbox.get("tmpfs_exec"), bool))
    # A tmpfs's pages are charged to the container's memory cgroup, and each path is a tmpfs of
    # tmpfs_bytes, so caps adding up past memory_bytes cap nothing; the size is what keeps the
    # executable scratch space a bounded one.
    memory = sandbox.get("memory_bytes")
    need(
        "submission.sandbox.tmpfs_bytes x len(tmpfs) in 1..memory_bytes",
        _positive_int(sandbox.get("tmpfs_bytes"))
        and _positive_number(memory)
        and sandbox["tmpfs_bytes"] * (len(paths) if paths else 1) <= memory,
    )
    need("submission.sandbox.user non-root", _non_root(sandbox.get("user")))
    for key in ("gpus", "memory_bytes", "cpus", "pids"):
        need(f"submission.sandbox.{key}>0", _positive_number(sandbox.get(key)))


def _weights_errors(model: Any, need) -> None:
    """A weights submission's `model`: the architecture whose code the validator owns, the one
    weights file a submission is, the files its repository may hold at all, the policy class that
    serves it, and the template digests it is held to (null until the template is generated)."""
    model = model if isinstance(model, dict) else {}
    need("submission.model mapping", bool(model))
    need("submission.model.architecture", _text(model.get("architecture")))
    weights = model.get("weights_file")
    need("submission.model.weights_file", _text(weights) and "/" not in str(weights))
    allowed = model.get("allowed_files")
    need(
        "submission.model.allowed_files list of names holding weights_file",
        isinstance(allowed, list)
        and all(_text(f) and "/" not in f for f in allowed)
        and weights in allowed,
    )
    need(
        "submission.model.policy module:Class",
        isinstance(model.get("policy"), str) and bool(MODULE_CLASS_RE.match(model["policy"])),
    )
    template = model.get("template")
    need("submission.model.template mapping", isinstance(template, dict))
    for key in ("cfg_sha256", "tensors_sha256"):
        value = (template or {}).get(key) if isinstance(template, dict) else None
        need(
            f"submission.model.template.{key} hex64|null",
            value is None or (isinstance(value, str) and bool(HEX64_RE.match(value))),
        )


def _crown_errors(where: str, crown: Any, need) -> None:
    """An optional crown rule: a paired test and its alpha in (0, 1)."""
    if crown is None:
        return
    need(f"{where} mapping", isinstance(crown, dict))
    if not isinstance(crown, dict):
        return
    need(f"{where}.paired_test in {list(PAIRED_TESTS)}", crown.get("paired_test") in PAIRED_TESTS)
    alpha = crown.get("alpha")
    need(f"{where}.alpha in (0,1)", _number(alpha) and 0 < alpha < 1)


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
        need(
            f"{gone} removed (v7); a weights submission's model is submission.model",
            gone not in doc,
        )

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
        env = s.get("environment")
        need(f"skills.{sid}.environment", isinstance(env, dict))
        _environment_errors(f"skills.{sid}.environment", env if isinstance(env, dict) else {}, need)
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
    _crown_errors("duel.crown", duel.get("crown"), need)

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
        need(f"tracks.{tid}.demonstration.withheld is empty", demo.get("withheld") == [])
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
        _crown_errors(f"tracks.{tid}.crown", t.get("crown"), need)
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
        if isinstance(entry, dict) and entry.get("size") is not None:
            track = tracks.get(tid) if isinstance(tracks, dict) else None
            own = track.get("sizes") if isinstance(track, dict) else None
            need(
                f"baselines.{tid}.size in sizes",
                entry["size"] in (own if isinstance(own, dict) else sizes),
            )

    budgets = doc.get("budgets") or {}
    for key in BUDGETS:
        need(f"budgets.{key}>0", _positive_number(budgets.get(key)))
    # What a policy's budget leaves of its unit is the benchmark's own time: a budget that leaves
    # none lets a slow policy run every unit into the kill that voids it for both sides.
    policy, unit = budgets.get("policy_budget_seconds"), budgets.get("unit_wall_seconds")
    need(
        "budgets.policy_budget_seconds<unit_wall_seconds",
        not (_positive_number(policy) and _positive_number(unit)) or policy < unit,
    )
    # Optional: a hung simulator's watchdog, and how often a stalled unit is tried again.
    if budgets.get("stall_seconds") is not None:
        need("budgets.stall_seconds>0", _positive_number(budgets.get("stall_seconds")))
    retries = budgets.get("stall_retries")
    need(
        "budgets.stall_retries int>=0",
        retries is None
        or (isinstance(retries, int) and not isinstance(retries, bool) and retries >= 0),
    )

    submission = doc.get("submission") or {}
    kind = submission.get("kind", "code")
    need(f"submission.kind in {list(SUBMISSION_KINDS)}", kind in SUBMISSION_KINDS)
    need("submission.policy_protocol:int", _positive_int(submission.get("policy_protocol")))
    need("submission.max_repo_bytes", _positive_int(submission.get("max_repo_bytes")))
    if kind == "weights":
        _weights_errors(submission.get("model"), need)
    else:
        _code_submission_errors(submission, need)

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

    def paired_alpha(self, track: str) -> float | None:
        """The sign test's alpha a duel of `track` must pass to move the crown (`duel.crown`, or
        the track's own `crown`); None where the spec sets none, and the margin alone decides."""
        crown = self.track(track).get("crown") or self.raw["duel"].get("crown")
        if not isinstance(crown, dict) or crown.get("alpha") is None:
            return None
        return float(crown["alpha"])

    # -- the rest
    @property
    def budgets(self) -> dict[str, Any]:
        return self.raw["budgets"]

    @property
    def stall(self) -> tuple[float | None, int]:
        """`(stall_seconds, stall_retries)`: the hung-simulator watchdog's window (None for no
        watchdog) and how many times a stalled unit or materialization is started again."""
        seconds = self.raw["budgets"].get("stall_seconds")
        return (
            None if seconds is None else float(seconds),
            int(self.raw["budgets"].get("stall_retries") or 0),
        )

    @property
    def submission(self) -> dict[str, Any]:
        return self.raw["submission"]

    @property
    def submission_kind(self) -> str:
        """`code` or `weights` (`SUBMISSION_KINDS`)."""
        return str(self.raw["submission"].get("kind", "code"))

    @property
    def model(self) -> dict[str, Any]:
        """A weights submission's `submission.model`; KeyError for a code submission's spec."""
        if self.submission_kind != "weights":
            raise KeyError("a code submission's spec has no submission.model")
        return dict(self.raw["submission"]["model"])

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
