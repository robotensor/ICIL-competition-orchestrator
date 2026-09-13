import importlib
import sys

# Simulator and model stacks a benchmark or a submission may use; none belongs in this process.
FORBIDDEN = {"sapien", "mujoco", "robosuite", "torch", "curobo"}


def test_the_package_imports_no_simulator():
    before = set(sys.modules)
    importlib.import_module("icil_orchestrator")
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    assert not loaded & FORBIDDEN
