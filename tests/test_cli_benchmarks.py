"""`vector-orchestrator benchmarks list|check`, against the fake benchmark installed for real."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from conftest import FAKE_PIN, FAKE_SITE, fake_spec_doc
from vector_orchestrator.benchmarks.check import check_benchmark
from vector_orchestrator.cli import main


def script() -> str:
    path = Path(sysconfig.get_path("scripts")) / "vector-orchestrator"
    assert path.exists(), "install the package (pip install -e .) for the console script"
    return str(path)


def test_the_console_script_checks_the_fake_benchmark_with_no_problems(fake_spec):
    env = {**os.environ, "PYTHONPATH": str(FAKE_SITE)}
    done = subprocess.run(
        [script(), "--spec", str(fake_spec.path), "benchmarks", "check", "fake"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "problem:" not in done.stdout
    assert done.stdout.strip().splitlines()[-1] == "fake: ok"


def test_check_reports_the_fake_benchmark_clean(fake_installed, fake_spec):
    report = check_benchmark(fake_spec, "fake")
    assert report.problems == []
    assert report.info["embodiment"][:2] == ["franka-panda", "franka-panda"]
    assert "wheel sha256 not pinned in spec.json yet" in report.notes


def test_list_names_every_benchmark_without_importing_one(fake_installed, fake_spec, capsys):
    assert main(["--spec", str(fake_spec.path), "benchmarks", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["name"], r["distribution"], r["loaded"]) for r in rows] == [
        ("fake", "vector-fake-benchmark", False)
    ]
    assert "vector_fake_benchmark" not in sys.modules


def test_check_refuses_another_api_version_with_the_reason(
    install_distribution, spec_doc, write_spec, capsys
):
    install_distribution(
        "vector-variant-benchmark",
        entry_points={"fake": "vector_variant_v2"},
        modules={
            "vector_variant_v2": (
                "import importlib.util\n"
                "s = importlib.util.spec_from_file_location("
                f"'vector_variant_base', {str(FAKE_SITE / 'vector_fake_benchmark' / '__init__.py')!r})\n"
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "class V2(m.FakeBenchmark):\n    api_version = 2\n"
                "BENCHMARK = V2()\n"
            )
        },
    )
    spec = write_spec(
        fake_spec_doc(spec_doc, {**FAKE_PIN, "distribution": "vector-variant-benchmark"})
    )
    assert main(["--spec", str(spec.path), "benchmarks", "check", "fake"]) == 1
    out = capsys.readouterr().out
    assert "problem: api_version: speaks 2, this orchestrator speaks 1" in out
    assert out.strip().endswith("fake: REFUSED")


@pytest.mark.parametrize(
    "mutation, problem",
    [
        (
            "    def catalogue(self):\n        return {'suites': {}, 'categories': {}}\n",
            "franka_pick_and_place: suite 'franka_1arm' is not in its catalogue",
        ),
        (
            "    def derive_units(self, **kw):\n"
            "        import random\n"
            "        units = super().derive_units(**kw)\n"
            "        units[0]['instance_params']['scene_seed'] = random.random()\n"
            "        return units\n",
            "franka_pick_and_place: derive_units is not a pure function of its arguments",
        ),
        (
            "    def run_command(self, **kw):\n        return 'bench run'\n",
            "franka_pick_and_place: run_command: expected a non-empty list of strings",
        ),
        (
            "    def info(self):\n        return {'id': 'fake', 'api_version': 1, 'x': __import__('mujoco')}\n",
            "calling its pure methods imported mujoco",
        ),
        # A stub that returns nothing is as broken as one that raises.
        ("    def info(self):\n        pass\n", "info: returned NoneType, not a mapping"),
        ("    def catalogue(self):\n        pass\n", "catalogue: returned NoneType, not a mapping"),
        (
            "    def derive_units(self, **kw):\n        pass\n",
            "franka_pick_and_place: derive_units: returned NoneType, not a list",
        ),
        (
            "    def materialize_command(self, **kw):\n        pass\n",
            "franka_pick_and_place: materialize_command: expected a non-empty list of strings, got None",
        ),
        (
            "    def run_command(self, **kw):\n        pass\n",
            "franka_pick_and_place: run_command: expected a non-empty list of strings, got None",
        ),
    ],
)
def test_check_catches_a_benchmark_that_would_fail_a_duel(
    install_distribution, spec_doc, write_spec, mutation, problem
):
    install_distribution(
        "vector-variant-benchmark",
        entry_points={"fake": "vector_variant_mutant"},
        modules={
            "vector_variant_mutant": (
                "import importlib.util\n"
                "s = importlib.util.spec_from_file_location("
                f"'vector_variant_base', {str(FAKE_SITE / 'vector_fake_benchmark' / '__init__.py')!r})\n"
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "class Mutant(m.FakeBenchmark):\n" + mutation + "BENCHMARK = Mutant()\n"
            ),
            "mujoco": "NAME = 'stand-in'\n",
        },
    )
    spec = write_spec(
        fake_spec_doc(spec_doc, {**FAKE_PIN, "distribution": "vector-variant-benchmark"})
    )
    try:
        report = check_benchmark(spec, "fake")
    finally:
        sys.modules.pop("mujoco", None)
    assert any(p.startswith(problem) for p in report.problems), report.problems


def test_check_hands_run_command_an_address_in_the_wire_form(
    install_distribution, spec_doc, write_spec
):
    """The policy wire listens on a Unix socket path or host:port (vector_policy.wire), so a plugin
    that follows the contract and checks the address must pass `check`, not be failed by a URL."""
    install_distribution(
        "vector-variant-benchmark",
        entry_points={"fake": "vector_variant_strict"},
        modules={
            "vector_variant_strict": (
                "import importlib.util, os\n"
                "s = importlib.util.spec_from_file_location("
                f"'vector_variant_base', {str(FAKE_SITE / 'vector_fake_benchmark' / '__init__.py')!r})\n"
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "class Strict(m.FakeBenchmark):\n"
                "    def run_command(self, *, policy_address, **kw):\n"
                "        host, _, port = policy_address.rpartition(':')\n"
                "        if not (os.path.isabs(policy_address) or (host and port.isdigit())):\n"
                "            raise ValueError(f'{policy_address!r} is not a socket path or host:port')\n"
                "        return super().run_command(policy_address=policy_address, **kw)\n"
                "BENCHMARK = Strict()\n"
            )
        },
    )
    spec = write_spec(
        fake_spec_doc(spec_doc, {**FAKE_PIN, "distribution": "vector-variant-benchmark"})
    )
    assert check_benchmark(spec, "fake").problems == []


def test_check_on_the_shipped_spec_says_robotwin_is_not_installed(spec, capsys):
    assert main(["benchmarks", "check", "robotwin"]) == 1
    out = capsys.readouterr().out
    assert "problem: not installed: install robotensor-benchmark-robotwin" in out


def test_python_dash_m_runs_the_same_cli(fake_spec):
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "vector_orchestrator",
            "--spec",
            str(fake_spec.path),
            "benchmarks",
            "list",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(FAKE_SITE)},
    )
    assert done.returncode == 0 and done.stdout.startswith("fake"), done.stderr
