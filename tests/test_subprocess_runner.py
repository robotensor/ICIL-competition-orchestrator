"""Driving a benchmark this process cannot import.

The fake benchmark's `run_command` is a real subprocess (`icil_fake_benchmark/command.py`) writing a
real result file, so the path under test is the one a duel takes - not a mock of it.
"""

from __future__ import annotations

import json
import subprocess
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


def test_the_side_wall_clock_voids_the_units_left_when_it_runs_out(fake, tmp_path):
    """`budgets.side_wall_seconds` bounds a side: once it is spent, the units not yet started are
    void as timed out rather than run, and the loop still reports every one of them."""
    seen = []
    outcomes = runner.run_units(
        fake,
        [unit(0, "hang"), unit(1), unit(2)],
        work_root=tmp_path / "units",
        policy_address="/tmp/icil-test-policy.sock",
        authkey_env=AUTHKEY_ENV,
        timeout_s=1.0,
        env={"PATH": "/usr/bin:/bin", AUTHKEY_ENV: AUTHKEY},
        deadline=time.monotonic() + 0.5,
        on_outcome=lambda u, o: seen.append(u["unit_id"]),
    )
    assert seen == ["fp-000", "fp-001", "fp-002"]
    first, *rest = outcomes
    assert first.error == "fake: unit exceeded its 1s budget"
    for outcome in rest:
        assert outcome.void and outcome.error == "fake: the side ran out of its wall-clock budget"
    assert not (tmp_path / "units" / "fp-001").exists(), "a unit past the deadline was run"


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


def _spawner(pid_file, then: str) -> list[str]:
    """A command that forks a long-lived child (a renderer, say), records its pid, then `then`."""
    return [
        sys.executable,
        "-c",
        "import subprocess,sys,time;"
        "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
        f"open({str(pid_file)!r},'w').write(str(c.pid));" + then,
    ]


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    return not _alive(pid)


@pytest.mark.parametrize("then, code", [("sys.exit(3)", 3), ("sys.exit(0)", 0)])
def test_a_unit_that_exits_takes_its_children_with_it(tmp_path, then, code):
    """A crash is the commonest way a simulator unit ends; what it forked must not hold the GPU
    while the next unit runs."""
    pid_file = tmp_path / "child.pid"
    done = runner.run_argv(
        _spawner(pid_file, then),
        env={"PATH": "/usr/bin:/bin"},
        timeout_s=30,
        log_path=tmp_path / "l",
    )
    assert done.returncode == code and not done.timed_out
    assert _gone(int(pid_file.read_text())), "the benchmark's child outlived its unit"


def test_an_interrupted_orchestrator_does_not_leave_the_benchmark_running(tmp_path):
    """The benchmark runs in its own session, so the terminal's Ctrl-C never reaches it; the
    orchestrator has to take it down on the way out."""
    import signal

    pid_file = tmp_path / "child.pid"

    def interrupt(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 1.5)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run_argv(
                _spawner(pid_file, "time.sleep(60)"),
                env={"PATH": "/usr/bin:/bin"},
                timeout_s=60,
                log_path=tmp_path / "l",
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert _gone(int(pid_file.read_text())), "the benchmark survived the interrupt"


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


def test_by_default_the_benchmark_sees_an_allow_listed_environment(fake, tmp_path, monkeypatch):
    """The benchmark parses a hostile policy's replies; it has no use for the credentials that
    publish results, and must not hold them."""
    monkeypatch.setenv("HF_TOKEN", "hf_publish_secret")
    monkeypatch.setenv("ICIL_LIVE_TOKEN", "live_secret")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv(AUTHKEY_ENV, AUTHKEY)

    class Snoop(type(fake)):
        def run_command(self, *, out_dir, **kw):
            code = (
                "import json, os\n"
                f"open({out_dir + '/env.json'!r}, 'w').write(json.dumps(dict(os.environ)))\n"
                f"open({out_dir + '/result.json'!r}, 'w').write('{{\"success\": true}}')\n"
            )
            return [sys.executable, "-c", code]

    outcome = runner.run_unit(
        Snoop(),
        unit(0),
        prompt="/prompts/p.npz",
        out_dir=tmp_path / "u",
        policy_address="/tmp/icil-test-policy.sock",
        authkey_env=AUTHKEY_ENV,
        timeout_s=30,
    )
    assert not outcome.void, outcome.error
    seen = json.loads((tmp_path / "u" / "env.json").read_text())
    assert "HF_TOKEN" not in seen and "ICIL_LIVE_TOKEN" not in seen
    assert seen[AUTHKEY_ENV] == AUTHKEY and seen["PATH"] and seen["LC_ALL"] == "C.UTF-8"
    kept = runner.benchmark_environment({"PATH": "/bin", "MUJOCO_GL": "egl", "X": "1"}, "K", ("X",))
    assert kept == {"PATH": "/bin", "MUJOCO_GL": "egl", "X": "1"}


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


def test_non_finite_numbers_in_a_result_never_stop_the_duel(fake, tmp_path):
    """JSON parsers accept NaN and Infinity; `int(nan)` raises. One such result must not lose the
    units after it, nor carry a NaN on to a record that cannot be signed."""

    class NaN(type(fake)):
        def run_command(self, *, out_dir, **kw):
            body = '{"success": true, "void": false, "steps": NaN, "progress": Infinity, "metric": -Infinity}'
            code = f"open({out_dir + '/result.json'!r}, 'w').write({body!r})"
            return [sys.executable, "-c", code]

        def read_result(self, *, out_dir):
            return runner.read_result_file(out_dir)

    outcomes = run(NaN(), [unit(0), unit(1)], tmp_path)
    assert len(outcomes) == 2
    for outcome in outcomes:
        assert (outcome.success, outcome.void) == (True, False)
        assert (outcome.steps, outcome.progress, outcome.metric) == (None, None, None)
    assert runner.outcome_from({"success": False, "steps": 1e400}).steps is None


def test_a_result_that_cannot_be_read_into_an_outcome_is_void(fake, tmp_path):
    class Odd(dict):
        def items(self):
            raise RuntimeError("not today")

    class Weird(type(fake)):
        def read_result(self, *, out_dir):
            return Odd(success=True, void=False)

    (outcome,) = run(Weird(), [unit(0)], tmp_path)
    assert outcome.void and outcome.error.startswith("fake: unusable result: RuntimeError")


def test_the_log_tail_reads_only_the_end_of_a_huge_log(tmp_path):
    """Nothing bounds what a benchmark prints; a unit's reason must not need the whole log in RAM.
    Run under a 1 GiB address-space limit against a 4 GiB (sparse) log."""
    log = tmp_path / "benchmark.log"
    with open(log, "wb") as fh:
        fh.truncate(4 << 30)
        fh.seek(0, 2)
        fh.write(b"\nthe simulator lost the GPU\n")
    code = (
        "import resource, sys\n"
        "resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))\n"
        "from pathlib import Path\n"
        "from icil_orchestrator.benchmarks.subprocess_runner import _tail\n"
        f"print(_tail(Path({str(log)!r})))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip().endswith("the simulator lost the GPU")


def test_read_result_file_never_raises(tmp_path):
    assert runner.read_result_file(tmp_path)["void"] is True
    (tmp_path / "result.json").write_text("[1, 2]")
    assert runner.read_result_file(tmp_path)["error"] == "unreadable result.json"
