import subprocess
import sys

import vector_policy

# What a competitor's image or a benchmark process may hold; none of it belongs in this distribution.
FORBIDDEN = {"sapien", "mujoco", "robosuite", "torch", "curobo", "vector_orchestrator", "yaml"}


def loaded_by(statement: str) -> set[str]:
    """Top-level modules, loaded from a file, that a fresh interpreter loads to run `statement`.

    Modules with no file are aliases and runtime bookkeeping (`__mp_main__`, Cython's).
    """
    code = (
        f"import sys; before = set(sys.modules); {statement}; "
        "print(' '.join(sorted({m.split('.')[0] for m in set(sys.modules) - before "
        "if getattr(sys.modules[m], '__file__', None)})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_importing_the_package_loads_no_simulator_model_stack_or_yaml():
    assert not loaded_by("import vector_policy") & FORBIDDEN


def test_a_benchmark_importing_the_client_loads_numpy_and_the_standard_library_only():
    loaded = loaded_by("import vector_policy.client")
    assert not loaded & FORBIDDEN
    assert loaded - set(sys.stdlib_module_names) <= {"numpy", "vector_policy"}


def test_the_public_names_are_exported():
    for name in ("Policy", "PolicyUnavailable", "ManifestError", "WireError", "ACTION_TYPES"):
        assert hasattr(vector_policy, name)
    assert vector_policy.__version__
