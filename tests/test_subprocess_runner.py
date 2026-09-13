"""Driving a benchmark this process cannot import.

The fake benchmark's `run_command` is a real subprocess (`icil_fake_benchmark/command.py`) writing a
real result file, so the path under test is the one a duel takes - not a mock of it.
"""

from __future__ import annotations

import sys
import time

import pytest

from icil_orchestrator.benchmarks import subprocess_runner as runner

AUTHKEY_ENV = "ICIL_TEST_POLICY_AUTHKEY"
AUTHKEY = "00112233445566778899aabbccddeeff"


@pytest.fixture
def fake(fake_installed):
    import icil_fake_benchmark

    return icil_fake_benchmark.BENCHMARK


def unit(n: int, behaviour: str = "succeed", prompt: str = "/prompts/p.npz") -> dict:
    return {
        "unit_id": f"fp-{n:03d}",
        "skill": "franka_pick_and_place",
        "task": "place_cube_plate",
        "instance_params": {"scene_seed": 7, "embodiment": ["franka-panda", "franka-panda", 0.6]},
        "prompt": prompt,
        "fake_behaviour": behaviour,
    }


def run(fake, units, tmp_path, timeout_s=30.0, env=None):
    seen = []
    outcomes = runner.run_units(
        fake,
        units,
        work_root=tmp_path / "units",
        policy_address="unix:///tmp/icil-test-policy.sock",
        authkey_env=AUTHKEY_ENV,
        timeout_s=timeout_s,
        env={"PATH": "/usr/bin:/bin", AUTHKEY_ENV: AUTHKEY} if env is None else env,
        on_outcome=lambda u, o: seen.append((u["unit_id"], o)),
    )
    assert [s[0] for s in seen] == [u["unit_id"] for u in units]
    return outcomes


def test_a_unit_runs_in_a_subprocess_and_its_result_comes_back(fake, tmp_path):
    (outcome,) = run(fake, [unit(0)], tmp_path)
    assert (outcome.success, outcome.void, outcome.steps, outcome.error) == (True, False, 5, None)
    assert outcome.progress == 1.0 and outcome.wall_s > 0
    assert outcome.clip == str(tmp_path / "units" / "fp-000" / "evaluation.mp4")
    assert "prompt_sha256" in outcome.extra, "what the benchmark adds is kept, not dropped"


def test_a_failed_episode_is_a_loss_not_a_void(fake, tmp_path):
    (outcome,) = run(fake, [unit(0, "fail")], tmp_path)
    assert (outcome.success, outcome.void) == (False, False)


def test_crash_timeout_and_no_result_are_void_with_the_reason_and_the_rest_still_run(
    fake, tmp_path
):
    units = [
        unit(0, "crash"),
        unit(1, "hang"),
        unit(2, "silent"),
        unit(3, "garbage"),
        unit(4, "succeed"),
    ]
    started = time.monotonic()
    outcomes = run(fake, units, tmp_path, timeout_s=2.0)
    assert time.monotonic() - started < 20, "a hung unit held up the rest"

    crash, hang, silent, garbage, fine = outcomes
    assert crash.void and crash.success is None
    assert (
        crash.error.startswith("fake: exited 3: ") and "the simulator lost the GPU" in crash.error
    )
    assert hang.void and hang.error == "fake: unit exceeded its 2s budget"
    assert silent.void and silent.error.startswith("fake: exited 0 but wrote no result.json")
    assert garbage.void and garbage.error == "fake: unreadable result.json"
    assert (fine.success, fine.void) == (True, False)


def test_a_silent_run_is_void_even_where_an_earlier_command_left_a_result(fake, tmp_path):
    """The directory a unit ran in before - an earlier attempt, or the materialize command, which
    writes result.json too - must not lend its result or clip to a run that wrote neither."""
    out = tmp_path / "units" / "fp-000"
    materialize = fake.materialize_command(
        unit={"task": "place_cube_plate", "instance_params": {"scene_seed": 7}}, out_dir=str(out)
    )
    assert (
        runner.run_argv(
            materialize, env={"PATH": "/usr/bin:/bin"}, timeout_s=30, log_path=tmp_path / "m.log"
        ).returncode
        == 0
    )
    assert (out / "result.json").is_file()
    (silent,) = run(fake, [unit(0, "silent")], tmp_path)
    assert silent.void and silent.error.startswith("fake: exited 0 but wrote no result.json")

    (first,) = run(fake, [unit(0, "succeed")], tmp_path)
    assert first.success and first.clip
    (again,) = run(fake, [unit(0, "silent")], tmp_path)
    assert again.void and again.success is None and again.clip is None


def test_a_hung_unit_is_killed_with_its_children(tmp_path):
    """The whole process group goes, so a forked renderer cannot keep the GPU."""
    pid_file = tmp_path / "child.pid"
    argv = [
        sys.executable,
        "-c",
        "import subprocess,sys,time;"
        "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
        f"open({str(pid_file)!r},'w').write(str(c.pid));"
        "time.sleep(60)",
    ]
    done = runner.run_argv(
        argv, env={"PATH": "/usr/bin:/bin"}, timeout_s=2, log_path=tmp_path / "l"
    )
    assert done.timed_out and done.returncode is None and done.wall_s < 10
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(child):
        time.sleep(0.05)
    assert not _alive(child), "the benchmark's child process outlived its unit"


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def test_a_command_that_cannot_start_is_void(tmp_path, fake):
    class Missing(type(fake)):
        def run_command(self, **kw):
            return ["/nonexistent/benchmark"]

    (outcome,) = run(Missing(), [unit(0)], tmp_path)
    assert outcome.void and outcome.error.startswith("fake: could not start the benchmark")


def test_the_authkey_travels_by_variable_name_only(fake, tmp_path):
    """The fake's command exits 4 when the variable is missing, so success proves it was read
    from the environment the runner passed."""
    (outcome,) = run(fake, [unit(0)], tmp_path)
    assert not outcome.void
    argv = fake.run_command(
        unit=unit(0),
        prompt="p",
        out_dir="o",
        policy_address="a",
        authkey_env=AUTHKEY_ENV,
    )
    assert AUTHKEY not in " ".join(argv) and AUTHKEY_ENV in argv


def test_a_plugin_that_puts_the_key_on_the_command_line_is_not_run(fake, tmp_path):
    class Leaky(type(fake)):
        def run_command(self, **kw):
            return [sys.executable, "-c", "raise SystemExit(0)", "--key", AUTHKEY]

    (outcome,) = run(Leaky(), [unit(0)], tmp_path)
    assert outcome.void and "put the policy authkey on the command line" in outcome.error
    assert not (tmp_path / "units" / "fp-000" / "benchmark.log").exists(), "it was run"


def test_no_authkey_in_the_environment_is_a_harness_fault(fake, tmp_path):
    (outcome,) = run(fake, [unit(0)], tmp_path, env={"PATH": "/usr/bin:/bin"})
    assert outcome.void and outcome.error == (
        f"fake: no policy authkey in ${AUTHKEY_ENV} for the benchmark subprocess"
    )


def test_a_unit_with_no_materialized_prompt_is_void_rather_than_run(fake, tmp_path):
    (outcome,) = run(fake, [unit(0, prompt="")], tmp_path)
    assert outcome.void and outcome.error == "fake: the unit carries no materialized prompt"


def test_a_plugin_that_raises_voids_its_unit_rather_than_the_duel(fake, tmp_path):
    class Exploding(type(fake)):
        def run_command(self, **kwargs):
            raise RuntimeError("boom")

    outcomes = run(Exploding(), [unit(0), unit(1)], tmp_path)
    assert [o.error for o in outcomes] == ["fake: run_command failed: RuntimeError: boom"] * 2


def test_success_and_void_cannot_disagree():
    """`success is None` exactly when void; reconciled toward void, because the alternative is a
    wrong score rather than an error."""
    assert runner.outcome_from({"success": None, "void": False}).void is True
    both = runner.outcome_from({"success": True, "void": True, "error": "scene drifted"})
    assert both.void is True and both.success is None and both.error == "scene drifted"
    assert runner.outcome_from({"success": "yes"}).void is True


def test_a_result_carrying_only_what_the_abi_requires_still_makes_an_outcome():
    out = runner.outcome_from({"success": False, "void": False, "steps": 3, "error": None})
    assert (out.success, out.void, out.steps, out.error) == (False, False, 3, None)
    assert out.progress is None and out.metric is None and out.extra == {}


def test_read_result_file_never_raises(tmp_path):
    assert runner.read_result_file(tmp_path)["void"] is True
    (tmp_path / "result.json").write_text("[1, 2]")
    assert runner.read_result_file(tmp_path)["error"] == "unreadable result.json"
