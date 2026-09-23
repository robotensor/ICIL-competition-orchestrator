import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import FAKE_SITE

# Simulator and model stacks a benchmark or a submission may use; none belongs in this process.
# `vector_fake_simulator` is the fake benchmark's stand-in for one.
FORBIDDEN = {"sapien", "mujoco", "robosuite", "torch", "curobo", "vector_fake_simulator"}


def test_the_package_imports_no_simulator():
    before = set(sys.modules)
    importlib.import_module("vector_orchestrator")
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    assert not loaded & FORBIDDEN


PROBE = """
import json, sys
import vector_orchestrator.benchmarks
from vector_orchestrator.benchmarks.check import check_benchmark
from vector_orchestrator.benchmarks.units import plugin_units
from vector_orchestrator.spec import load_spec_file

spec = load_spec_file(sys.argv[1])
report = check_benchmark(spec, "fake")
units = plugin_units(spec, "franka_1arm", "d" * 64, "smoke")
print(json.dumps({
    "ok": report.ok,
    "units": len(units),
    "modules": sorted({m.split(".")[0] for m in sys.modules}),
}))
"""


def test_importing_and_driving_the_benchmarks_package_loads_no_simulator(fake_spec):
    """With the fake benchmark installed, importing `vector_orchestrator.benchmarks`, loading the
    plugin, checking it and deriving a duel's units - in a fresh interpreter - imports neither a
    real simulator stack nor the fake benchmark's stand-in for one."""
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(fake_spec.path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(FAKE_SITE)},
        check=True,
    )
    probe = json.loads(done.stdout)
    assert probe["ok"] and probe["units"] == 3
    assert "vector_fake_benchmark" in probe["modules"], "the plugin was not really loaded"
    assert not set(probe["modules"]) & FORBIDDEN


def test_the_local_state_a_run_writes_is_git_ignored():
    """`store init` and `queue` write into the checkout by default; the signing key committed to a
    public repository would let anyone forge the store's records."""
    from vector_orchestrator.cli import DEFAULT_KEY

    root = Path(__file__).resolve().parents[1]
    ignored = {
        line.strip() for line in (root / ".gitignore").read_text().splitlines() if line.strip()
    }
    assert f"/{DEFAULT_KEY.split('/')[0]}/" in ignored
    assert "/queue/" in ignored
