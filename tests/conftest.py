from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

from icil_orchestrator.spec import load_spec, load_spec_file

TRACK = "franka_1arm"

#: A site directory holding the fake benchmark's package and its `.dist-info`, so prepending it
#: to `sys.path` installs the plugin for real: `importlib.metadata` finds its entry point exactly
#: as it finds a pip-installed one.
FAKE_SITE = Path(__file__).resolve().parent / "fake_benchmark"
FAKE_PIN = {
    "distribution": "icil-fake-benchmark",
    "api_version": 1,
    "version": "0.1.0",
    "wheel_sha256": None,
}
#: Modules the fake distributions bring; dropped after each test so none leaks into the next.
FAKE_MODULE_PREFIXES = ("icil_fake_benchmark", "icil_fake_simulator", "icil_variant_")


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
def sandbox_spec(spec, tmp_path):
    """The contract, with the sandbox user this process can hand a directory to: root can give
    it to the spec's user, anyone else only to themselves. The limits are the spec's own."""
    if os.getuid() == 0:
        return spec
    doc = json.loads(spec.path.read_text())
    doc["submission"]["sandbox"]["user"] = f"{os.getuid()}:{os.getgid()}"
    (tmp_path / "sandbox-spec.json").write_text(json.dumps(doc))
    return load_spec_file(tmp_path / "sandbox-spec.json")


@pytest.fixture(autouse=True)
def shared_mounts(request, monkeypatch):
    """The pure suite mounts no filesystem: the shared directory's tmpfs is recorded here, as
    `("mount", directory, uid, gid)` and `("umount", directory)`, instead of mounted. A container
    test mounts it for real."""
    calls: list[tuple] = []
    if request.node.get_closest_marker("container") is None:
        monkeypatch.setattr(
            "icil_orchestrator.submissions.container.mount_shared_dir",
            lambda directory, uid, gid: calls.append(("mount", Path(directory), uid, gid)),
        )
        monkeypatch.setattr(
            "icil_orchestrator.submissions.container.unmount_shared_dir",
            lambda directory: calls.append(("umount", Path(directory))),
        )
    return calls


@pytest.fixture
def write_spec(tmp_path):
    """Save a (modified) contract and load it through the validator."""

    def write(doc, name="spec.json"):
        path = tmp_path / name
        path.write_text(json.dumps(doc, indent=1))
        return load_spec_file(path)

    return write


def fake_spec_doc(doc: dict, pin: dict | None = None) -> dict:
    """The shipped contract with every skill moved onto the fake benchmark, which is what a test
    without RoboTwin runs the franka_1arm track on."""
    doc = copy.deepcopy(doc)
    doc["benchmarks"] = {"fake": dict(FAKE_PIN if pin is None else pin)}
    for skill in doc["skills"].values():
        skill["benchmark"] = "fake"
    return doc


@pytest.fixture
def fake_spec(spec_doc, write_spec):
    return write_spec(fake_spec_doc(spec_doc), name="fake-spec.json")


@pytest.fixture(autouse=True)
def _forget_fake_modules():
    yield
    for name in [m for m in sys.modules if m.startswith(FAKE_MODULE_PREFIXES)]:
        del sys.modules[name]


@pytest.fixture
def fake_installed(monkeypatch):
    """The fake benchmark distribution, installed on `sys.path`."""
    monkeypatch.syspath_prepend(str(FAKE_SITE))
    return FAKE_SITE


@pytest.fixture
def install_distribution(tmp_path, monkeypatch):
    """Install a throwaway distribution: a `.dist-info` with entry points, its modules, and
    optionally the PEP 610 `direct_url.json` an installer would have written."""

    def install(
        name: str,
        *,
        entry_points: dict[str, str],
        modules: dict[str, str],
        version: str = "0.1.0",
        direct_url: dict | None = None,
    ) -> Path:
        site = tmp_path / f"site-{name}"
        info = site / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        lines = ["[icil.benchmarks]"] + [f"{k} = {v}" for k, v in entry_points.items()]
        (info / "entry_points.txt").write_text("\n".join(lines) + "\n")
        if direct_url is not None:
            (info / "direct_url.json").write_text(json.dumps(direct_url))
        for module, code in modules.items():
            (site / f"{module}.py").write_text(code)
        monkeypatch.syspath_prepend(str(site))
        return site

    return install
