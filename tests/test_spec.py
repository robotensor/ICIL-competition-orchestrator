import pytest

from icil_orchestrator.spec import SpecError, load_schema, validate_spec

#: The categories in robotensor/ICIL-robotwin-benchmark `src/robotwin_icil/tasks.yml`. A skill's
#: category is passed to the benchmark's `derive_units`, so a typo would derive nothing.
ROBOTWIN_CATEGORIES = {
    "pick_and_place",
    "stacking",
    "press_push",
    "open_close",
    "insertion",
    "bimanual",
    "articulated",
}


def test_the_shipped_contract_is_one_franka_track_on_robotwin(spec, track):
    assert validate_spec(spec.raw) == []
    assert spec.version == 7
    assert spec.tracks == (track,) and spec.sole_track == track
    assert spec.skills(track) == ("franka_pick_and_place", "franka_stacking", "franka_press_push")
    assert spec.benchmarks_of(track) == ("robotwin",)
    for skill in spec.skills(track):
        assert spec.suite(skill) == "franka_1arm"
        assert spec.category(skill) in ROBOTWIN_CATEGORIES
        assert spec.env(skill)["embodiment"][:2] == ["franka-panda", "franka-panda"]
        assert spec.env(skill)["action_dims"] == {"qpos": 16, "ee": 16}
    assert spec.benchmark_pin("robotwin")["distribution"] == "robotwin-icil-competition"
    assert "PROVISIONAL" in spec.raw["_skills_comment"]
    assert spec.units_per_side(track, "smoke") == 3 * spec.units_per_skill(track, "smoke")
    assert spec.size_of(track, "bogus") == spec.default_size(track)
    assert 0 <= spec.score_margin(track) <= 100 and 0 <= spec.max_void_fraction(track) <= 1
    assert spec.store["schema"] == 4 and spec.live["schema"] == 4
    assert spec.baseline(track) is None
    assert len(spec.fingerprint) == 64


def test_the_weights_era_blocks_are_gone(spec):
    assert "model" not in spec.raw and "pools" not in spec.raw
    assert spec.submission["sandbox"]["network"] == "none"
    for skill in spec.all_skills:
        assert not {"architecture", "simulator", "tasks"} & set(spec.skill(skill))


def test_the_dashboard_accepts_the_contract(spec):
    """The checks `validateSpec` in robofluent/ICIL-competition-dashboard `lib/spec.ts` (branch
    milestone-two-contests) and `scripts/sync-spec.mjs` make at build time. A contract failing
    them would not build the site."""
    raw = spec.raw
    assert "track" not in raw and isinstance(raw["tracks"], dict) and raw["tracks"]
    claimed = []
    for tid, t in raw["tracks"].items():
        for key in ("short", "title", "blurb", "slug"):
            assert isinstance(t[key], str) and t[key], (tid, key)
        assert isinstance(t["k_demos"], int) and t["k_demos"] >= 1
        assert isinstance(t["skills"], list) and t["skills"]
        claimed += t["skills"]
        assert "video" in t["demonstration"]["modalities"]
    assert sorted(claimed) == sorted(raw["skills"]) and len(set(claimed)) == len(claimed)
    codes = set()
    for s in raw["skills"].values():
        assert isinstance(s["title"], str) and s["title"]
        assert len(s["code"]) == 2 and s["code"].islower() and s["code"] not in codes
        codes.add(s["code"])
        assert "perturbations" not in s and isinstance(s["environment"], dict)
        assert isinstance(s["max_steps"], int) and s["max_steps"] >= 1
    duel = raw["duel"]
    assert duel["default_size"] in duel["sizes"]
    assert all(v["units_per_skill"] >= 1 for v in duel["sizes"].values())
    assert 0 <= duel["score_margin"] <= 100
    assert isinstance(raw["store"]["index_lines_per_part"], int)
    assert isinstance(raw["store"]["schema"], int) and isinstance(raw["live"]["schema"], int)
    for tid in raw["tracks"]:
        assert tid in raw["baselines"]
        b = raw["baselines"][tid]
        assert b is None or isinstance(b.get("repo"), str)


def test_the_store_schema_loads_and_names_every_published_shape():
    schema = load_schema()
    assert {"IndexRecord", "DuelEvent", "Head", "Manifest", "QueueSnapshot", "LiveFrame"} <= set(
        schema["$defs"]
    )


def test_validate_rejects_bad_specs(spec_doc, write_spec):
    def errors_after(mutate):
        import copy

        doc = copy.deepcopy(spec_doc)
        mutate(doc)
        return validate_spec(doc)

    assert any(
        "score_margin" in e for e in errors_after(lambda d: d["duel"].update(score_margin=101))
    )
    assert "skills non-empty" in errors_after(lambda d: d.update(skills={}))
    assert "skills.franka_stacking.code unique" in errors_after(
        lambda d: d["skills"]["franka_stacking"].update(code="fp")
    )
    assert "skills.franka_stacking.benchmark declared" in errors_after(
        lambda d: d["skills"]["franka_stacking"].update(benchmark="unity")
    )
    assert "skills.franka_stacking.architecture removed" in errors_after(
        lambda d: d["skills"]["franka_stacking"].update(architecture="bpp")
    )
    assert any("model removed" in e for e in errors_after(lambda d: d.update(model={})))
    assert "benchmarks.robotwin.api_version == 1" in errors_after(
        lambda d: d["benchmarks"]["robotwin"].update(api_version=2)
    )
    assert "benchmarks.robotwin.wheel_sha256 hex64|null" in errors_after(
        lambda d: d["benchmarks"]["robotwin"].update(wheel_sha256="abc")
    )
    assert "budgets.policy_budget_seconds>0" in errors_after(
        lambda d: d["budgets"].pop("policy_budget_seconds")
    )
    assert "budgets.policy_budget_seconds<unit_wall_seconds" in errors_after(
        lambda d: d["budgets"].update(policy_budget_seconds=d["budgets"]["unit_wall_seconds"])
    )
    doc = dict(spec_doc, duel=dict(spec_doc["duel"], default_size="gigantic"))
    with pytest.raises(SpecError, match="duel.default_size in sizes"):
        write_spec(doc)


def test_the_sandbox_cannot_be_loosened(spec_doc):
    """The sandbox is the whole of what an untrusted submission is held to."""
    import copy

    for key, value, message in (
        ("network", "bridge", "submission.sandbox.network == none"),
        ("read_only_root", False, "submission.sandbox.read_only_root"),
        ("user", "0:0", "submission.sandbox.user non-root"),
        ("user", "root", "submission.sandbox.user non-root"),
        # uid 0 spelled another way, and a root group.
        ("user", "00", "submission.sandbox.user non-root"),
        ("user", "0000:0000", "submission.sandbox.user non-root"),
        ("user", "1000:0", "submission.sandbox.user non-root"),
        ("user", "1000:root", "submission.sandbox.user non-root"),
        ("pids", 0, "submission.sandbox.pids>0"),
    ):
        doc = copy.deepcopy(spec_doc)
        doc["submission"]["sandbox"][key] = value
        assert message in validate_spec(doc), (key, value)


def test_the_scratch_tmpfs_is_capped_and_says_whether_it_runs_code(spec, spec_doc):
    """`/tmp` is where a policy's JIT caches compile and load code: an executable tmpfs, and a
    bounded one. The additive keys sit beside `tmpfs`, which keeps the shape the dashboard reads.
    Each path is one `docker run --tmpfs` takes as spelled and that hides nothing the container
    needs, and each gets `tmpfs_bytes`, so all of them together fit in `memory_bytes`."""
    import copy

    from icil_orchestrator.spec import SANDBOX_RESERVED_PATHS
    from icil_orchestrator.submissions.container import SOCKET_DIR
    from icil_orchestrator.submissions.image import SUBMISSION_DIR

    sandbox = spec.submission["sandbox"]
    memory = sandbox["memory_bytes"]
    assert sandbox["tmpfs"] == ["/tmp"] and sandbox["tmpfs_exec"] is True
    assert 0 < sandbox["tmpfs_bytes"] <= memory
    assert {SUBMISSION_DIR, SOCKET_DIR} <= set(SANDBOX_RESERVED_PATHS)
    paths = "submission.sandbox.tmpfs non-empty list of absolute paths"
    distinct = "submission.sandbox.tmpfs paths distinct"
    reserved = "submission.sandbox.tmpfs not / and not at or under "
    size = "submission.sandbox.tmpfs_bytes x len(tmpfs) in 1..memory_bytes"
    for changes, message in (
        ({"tmpfs": []}, paths),
        ({"tmpfs": "/tmp"}, paths),
        ({"tmpfs": ["tmp"]}, paths),
        # A colon starts docker's options and a comma separates them: either would smuggle some in.
        ({"tmpfs": ["/tmp:suid"]}, paths),
        ({"tmpfs": ["/tmp,dev"]}, paths),
        ({"tmpfs": ["/t mp"]}, paths),
        # The path as spelled is the path meant: nothing a normalisation would change.
        ({"tmpfs": ["/tmp/"]}, paths),
        ({"tmpfs": ["//tmp"]}, paths),
        ({"tmpfs": ["/tmp/./x"]}, paths),
        ({"tmpfs": ["/tmp/../submission"]}, paths),
        ({"tmpfs": ["/tmp", "/tmp"]}, distinct),
        # Docker refuses "/" and runc a tmpfs over /proc; one over the checkout empties it, one
        # over the socket's directory hides the socket from the host, and /sys and /dev are the
        # runtime's.
        ({"tmpfs": ["/"]}, reserved),
        ({"tmpfs": ["/tmp", "/proc"]}, reserved),
        ({"tmpfs": ["/sys/fs/cgroup"]}, reserved),
        ({"tmpfs": ["/dev/shm"]}, reserved),
        ({"tmpfs": ["/submission"]}, reserved),
        ({"tmpfs": ["/submission/cache"]}, reserved),
        ({"tmpfs": ["/run/icil"]}, reserved),
        ({"tmpfs_exec": "yes"}, "submission.sandbox.tmpfs_exec bool"),
        ({"tmpfs_exec": None}, "submission.sandbox.tmpfs_exec bool"),
        ({"tmpfs_bytes": 0}, size),
        ({"tmpfs_bytes": True}, size),
        ({"tmpfs_bytes": "8g"}, size),
        ({"tmpfs_bytes": 1.5}, size),
        ({"tmpfs_bytes": memory + 1}, size),
        # Each path is a tmpfs of tmpfs_bytes, all charged to the one memory cgroup.
        ({"tmpfs": ["/tmp", "/var/tmp"], "tmpfs_bytes": memory // 2 + 1}, size),
    ):
        doc = copy.deepcopy(spec_doc)
        doc["submission"]["sandbox"].update(changes)
        errors = validate_spec(doc)
        assert any(e.startswith(message) for e in errors), (changes, errors)
    for key in ("tmpfs_exec", "tmpfs_bytes"):
        doc = copy.deepcopy(spec_doc)
        del doc["submission"]["sandbox"][key]
        assert any(e.startswith(f"submission.sandbox.{key}") for e in validate_spec(doc)), key
    # A noexec scratch space is still a contract the orchestrator can honour, and so are several
    # paths that fit together: /run, under which the socket's directory is mounted on top, and a
    # name that only starts like the checkout's.
    doc = copy.deepcopy(spec_doc)
    doc["submission"]["sandbox"].update(tmpfs_exec=False, tmpfs_bytes=1 << 20)
    assert validate_spec(doc) == []
    doc["submission"]["sandbox"].update(
        tmpfs=["/tmp", "/run", "/submission-cache"], tmpfs_bytes=memory // 3
    )
    assert validate_spec(doc) == []


def test_a_skill_environment_is_the_shape_the_benchmark_and_dashboard_read(spec_doc):
    """The demonstration's shape lives here: RoboTwin's [left arm, right arm, distance], the clip
    cameras the dashboard lays out, and the action dimensions a policy must return."""
    import copy

    for path, value, message in (
        (("embodiment",), "franka-panda", "skills.franka_stacking.environment.embodiment"),
        (("embodiment",), ["franka-panda"], "skills.franka_stacking.environment.embodiment"),
        (("cameras",), "head_camera", "skills.franka_stacking.environment.cameras"),
        (("cameras",), [], "skills.franka_stacking.environment.cameras"),
        (("action_types",), ["qpos", "torque"], "skills.franka_stacking.environment.action_types"),
        (("action_types",), [], "skills.franka_stacking.environment.action_types"),
        (("action_dims",), {"qpos": "16"}, "skills.franka_stacking.environment.action_dims.qpos>0"),
        (("action_dims",), {"qpos": 16}, "skills.franka_stacking.environment.action_dims.ee>0"),
    ):
        doc = copy.deepcopy(spec_doc)
        doc["skills"]["franka_stacking"]["environment"][path[0]] = value
        assert any(e.startswith(message) for e in validate_spec(doc)), (path, value)


def test_only_a_view_the_orchestrator_can_honour_is_accepted(spec_doc):
    """Demonstration views beyond `sensorimotor` need the withholding that was not ported; a spec
    that asks for one would publish a view nothing enforces."""
    import copy

    doc = copy.deepcopy(spec_doc)
    doc["tracks"]["franka_1arm"]["demonstration"]["view"] = "video_only"
    assert "tracks.franka_1arm.demonstration.view" in validate_spec(doc)

    doc = copy.deepcopy(spec_doc)
    doc["tracks"]["franka_1arm"]["demonstration"]["withheld"] = ["actions"]
    assert "tracks.franka_1arm.demonstration.withheld is empty" in validate_spec(doc)


def test_every_skill_belongs_to_exactly_one_track(spec_doc):
    import copy

    orphan = copy.deepcopy(spec_doc)
    orphan["tracks"]["franka_1arm"]["skills"] = ["franka_stacking", "franka_press_push"]
    assert "tracks partition the skills" in validate_spec(orphan)

    twice = copy.deepcopy(spec_doc)
    twice["tracks"]["franka_1arm"]["skills"].append("franka_stacking")
    assert "tracks claim no skill twice" in validate_spec(twice)


def test_same_scene_may_not_claim_a_disjoint_prompt(spec_doc):
    import copy

    doc = copy.deepcopy(spec_doc)
    doc["tracks"]["franka_1arm"]["prompt_instance_disjoint"] = True
    assert "tracks.franka_1arm.prompt_instance_disjoint matches protocol" in validate_spec(doc)


def test_a_track_carries_its_own_duelling_constants(spec_doc, write_spec):
    import copy

    doc = copy.deepcopy(spec_doc)
    doc["tracks"]["franka_1arm"]["max_void_fraction"] = 0.25
    doc["tracks"]["franka_1arm"]["score_margin"] = 7.5
    loaded = write_spec(doc)
    assert loaded.max_void_fraction("franka_1arm") == 0.25
    assert loaded.score_margin("franka_1arm") == 7.5


def test_sole_track_refuses_to_guess(spec_doc, write_spec):
    import copy

    doc = copy.deepcopy(spec_doc)
    second = copy.deepcopy(doc["tracks"]["franka_1arm"])
    second.update(id="second", code="sc", slug="second", skills=["franka_press_push"])
    doc["tracks"]["franka_1arm"]["skills"] = ["franka_pick_and_place", "franka_stacking"]
    doc["tracks"]["second"] = second
    doc["baselines"]["second"] = None
    loaded = write_spec(doc)
    with pytest.raises(ValueError, match="say which one"):
        _ = loaded.sole_track
    assert loaded.track_of("franka_press_push") == "second"
