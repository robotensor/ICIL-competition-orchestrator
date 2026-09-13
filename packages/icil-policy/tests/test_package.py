import subprocess
import sys

import icil_policy

# What a competitor's image or a benchmark process may hold; none of it belongs in this distribution.
FORBIDDEN = {"sapien", "mujoco", "robosuite", "torch", "curobo", "icil_orchestrator", "yaml"}


def loaded_by(statement: str) -> set[str]:
    """Top-level modules a fresh interpreter loads to run `statement`."""
    code = (
        f"import sys; before = set(sys.modules); {statement}; "
        "print(' '.join(sorted({m.split('.')[0] for m in set(sys.modules) - before})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_importing_the_package_loads_no_simulator_model_stack_or_yaml():
    assert not loaded_by("import icil_policy") & FORBIDDEN


def test_the_public_names_are_exported():
    for name in ("Policy", "PolicyUnavailable", "ManifestError", "WireError", "ACTION_TYPES"):
        assert hasattr(icil_policy, name)
    assert icil_policy.__version__
