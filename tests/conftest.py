from __future__ import annotations

import json

import pytest

from icil_orchestrator.spec import load_spec, load_spec_file

TRACK = "franka_1arm"


@pytest.fixture(scope="session")
def spec():
    return load_spec()


@pytest.fixture(scope="session")
def track():
    return TRACK


@pytest.fixture
def spec_doc(spec):
    """A mutable copy of the shipped contract, so a rule is tested against the real spec."""
    return json.loads(spec.path.read_text())


@pytest.fixture
def write_spec(tmp_path):
    """Save a (modified) contract and load it through the validator."""

    def write(doc, name="spec.json"):
        path = tmp_path / name
        path.write_text(json.dumps(doc, indent=1))
        return load_spec_file(path)

    return write
