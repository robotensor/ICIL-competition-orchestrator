import importlib
import json
import os
import subprocess
import sys

from conftest import FAKE_SITE

# Simulator and model stacks a benchmark or a submission may use; none belongs in this process.
# `icil_fake_simulator` is the fake benchmark's stand-in for one.
FORBIDDEN = {"sapien", "mujoco", "robosuite", "torch", "curobo", "icil_fake_simulator"}


def test_the_package_imports_no_simulator():
    before = set(sys.modules)
    importlib.import_module("icil_orchestrator")
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    assert not loaded & FORBIDDEN


PROBE = """
import json, sys
import icil_orchestrator.benchmarks
from icil_orchestrator.benchmarks.check import check_benchmark
from icil_orchestrator.benchmarks.units import plugin_units
from icil_orchestrator.spec import load_spec_file

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
    """With the fake benchmark installed, importing `icil_orchestrator.benchmarks`, loading the
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
    assert "icil_fake_benchmark" in probe["modules"], "the plugin was not really loaded"
    assert not set(probe["modules"]) & FORBIDDEN
