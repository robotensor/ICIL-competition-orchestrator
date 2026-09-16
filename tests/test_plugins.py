"""Discovering a benchmark in its own distribution, and refusing the wrong one.

The fake benchmark is installed for real (a `.dist-info` on `sys.path`), so these tests exercise
`importlib.metadata` entry point discovery rather than a mock of it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import zipfile
from importlib import metadata

import pytest

from conftest import FAKE_PIN, FAKE_SITE, fake_spec_doc
from icil_orchestrator.benchmarks import plugins
from icil_orchestrator.benchmarks.plugins import BenchmarkRefused, discover, load

#: A plugin module that copies the fake benchmark from its file, without the fake's own
#: distribution on sys.path, and then breaks it in the way `{mutation}` says.
VARIANT = """
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "icil_variant_copy", {path!r}
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

class Variant(_module.FakeBenchmark):
{mutation}

BENCHMARK = Variant()
"""


def variant(mutation: str) -> str:
    return VARIANT.format(
        path=str(FAKE_SITE / "icil_fake_benchmark" / "__init__.py"), mutation=mutation
    )


def test_the_fake_benchmark_is_found_by_entry_point_and_loads(fake_installed, fake_spec):
    found = discover(fake_spec, load=True)
    fake = found["fake"]
    assert fake.ok, fake.problems
    assert (fake.distribution, fake.version, fake.entry_point) == (
        "icil-fake-benchmark",
        "0.1.0",
        "icil_fake_benchmark",
    )
    assert fake.benchmark.id == "fake"
    assert "wheel sha256 not pinned in spec.json yet" in fake.notes
    assert load(fake_spec, "fake") is fake.benchmark


def test_an_undeclared_benchmark_is_listed_but_never_imported(fake_installed, spec):
    found = discover(spec, load=True)
    assert found["fake"].problems == [
        "not declared in spec.json `benchmarks`; an undeclared benchmark is never imported"
    ]
    assert "icil_fake_benchmark" not in sys.modules
    with pytest.raises(BenchmarkRefused, match="not declared"):
        load(spec, "fake")


def test_a_declared_benchmark_that_is_absent_names_what_to_install(spec):
    robotwin = discover(spec)["robotwin"]
    assert robotwin.problems == [
        "not installed: install robotwin-icil-competition, which advertises 'robotwin' in the "
        "'icil.benchmarks' entry point group"
    ]


@pytest.mark.parametrize(
    "pin, problem",
    [
        (
            {**FAKE_PIN, "distribution": "someone-elses-benchmark"},
            "provided by distribution 'icil-fake-benchmark', but spec.json pins "
            "'someone-elses-benchmark'",
        ),
        ({**FAKE_PIN, "version": "0.2.0"}, "version 0.1.0 is not the pinned 0.2.0"),
        (
            {**FAKE_PIN, "wheel_sha256": "a" * 64},
            "the pinned wheel sha256 cannot be confirmed: the distribution records no wheel hash "
            "(install it from the pinned wheel file)",
        ),
    ],
)
def test_a_pin_mismatch_refuses_the_plugin_before_importing_it(
    fake_installed, spec_doc, write_spec, pin, problem
):
    spec = write_spec(fake_spec_doc(spec_doc, pin))
    fake = discover(spec, load=True)["fake"]
    assert problem in fake.problems
    assert fake.benchmark is None
    assert "icil_fake_benchmark" not in sys.modules, "a refused plugin was imported"


def test_the_wheel_sha256_is_checked_against_what_the_installer_recorded(
    install_distribution, spec_doc, write_spec
):
    wheel = hashlib.sha256(b"the wheel").hexdigest()
    install_distribution(
        "icil-variant-benchmark",
        entry_points={"fake": "icil_variant_ok"},
        modules={"icil_variant_ok": variant("    pass")},
        direct_url={
            "url": "file:///wheels/icil_variant_benchmark-0.1.0-py3-none-any.whl",
            "archive_info": {"hashes": {"sha256": wheel}},
        },
    )
    pin = {**FAKE_PIN, "distribution": "icil-variant-benchmark", "wheel_sha256": wheel}
    assert discover(write_spec(fake_spec_doc(spec_doc, pin)), load=True)["fake"].ok

    other = {**pin, "wheel_sha256": "b" * 64}
    fake = discover(write_spec(fake_spec_doc(spec_doc, other), name="other.json"), load=True)
    assert fake["fake"].problems == [
        f"installed from wheel sha256 {wheel[:16]}..., not the pinned {'b' * 16}..."
    ]


def test_the_wheel_hash_is_read_from_every_place_an_installer_leaves_it(tmp_path):
    class Dist:
        def __init__(self, doc):
            self.doc = doc

        def read_text(self, name):
            return None if self.doc is None else json.dumps(self.doc)

    digest = hashlib.sha256(b"wheel bytes").hexdigest()
    read = plugins.installed_wheel_sha256
    assert read(Dist({"url": "file:///x.whl", "archive_info": {"hash": f"sha256={digest}"}})) == (
        digest
    )
    assert read(Dist({"url": f"file:///x.whl#sha256={digest}", "archive_info": {}})) == digest
    assert read(Dist({"url": "file:///gone.whl", "archive_info": {}})) is None
    assert read(Dist(None)) is None

    # uv records the file it installed from but no hash; the file is hashed if it is still there
    # and its RECORD is what was installed.
    wheel = tmp_path / "x-0.1.0-py3-none-any.whl"
    info, record = _install_with_record(tmp_path / "site", "x", "x_mod", "X = 1\n", wheel.as_uri())
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("x_mod.py", "X = 1\n")
        zf.writestr(f"{info.name}/RECORD", "\n".join(record) + "\n")
    assert read(metadata.PathDistribution(info)) == plugins.sha256_file(wheel)


@pytest.mark.parametrize(
    "mutation, problem",
    [
        ("    api_version = 2", "api_version: speaks 2, this orchestrator speaks 1"),
        ("    run_command = None", "run_command: missing"),
        ("    id = 'elsewhere'", "icil_variant_bad is advertised as 'fake' but calls itself"),
        ("    import mujoco_stub_never_exists", "did not import: ModuleNotFoundError"),
    ],
)
def test_a_plugin_that_is_not_a_usable_benchmark_is_refused_with_the_reason(
    install_distribution, spec_doc, write_spec, mutation, problem
):
    install_distribution(
        "icil-variant-benchmark",
        entry_points={"fake": "icil_variant_bad"},
        modules={"icil_variant_bad": variant(mutation)},
    )
    spec = write_spec(
        fake_spec_doc(spec_doc, {**FAKE_PIN, "distribution": "icil-variant-benchmark"})
    )
    with pytest.raises(BenchmarkRefused) as refused:
        load(spec, "fake")
    assert problem in str(refused.value)


def test_a_plugin_that_imports_a_simulator_is_refused(install_distribution, spec_doc, write_spec):
    install_distribution(
        "icil-variant-benchmark",
        entry_points={"fake": "icil_variant_heavy"},
        modules={
            "icil_variant_heavy": "import mujoco\n" + variant("    pass"),
            "mujoco": "NAME = 'a stand-in for the real simulator'\n",
        },
    )
    spec = write_spec(
        fake_spec_doc(spec_doc, {**FAKE_PIN, "distribution": "icil-variant-benchmark"})
    )
    try:
        fake = discover(spec, load=True)["fake"]
    finally:
        sys.modules.pop("mujoco", None)
    assert "importing icil_variant_heavy imported mujoco; a plugin's pure half must not" in (
        fake.problems
    )


def test_two_distributions_claiming_one_name_are_refused(
    fake_installed, install_distribution, fake_spec
):
    install_distribution(
        "icil-variant-benchmark",
        entry_points={"fake": "icil_variant_twin"},
        modules={"icil_variant_twin": variant("    pass")},
    )
    fake = discover(fake_spec, load=True)["fake"]
    assert any("advertised by 2 entry points" in p for p in fake.problems)
    assert fake.benchmark is None


def test_a_module_shadowing_the_pinned_distribution_is_refused_before_import(
    install_distribution, spec_doc, write_spec, tmp_path, monkeypatch
):
    """The pin is checked on a distribution's metadata, but an entry point imports by module name:
    a same-named module earlier on sys.path (the cwd under `python -m`, a checkout) would be what
    runs. It must be refused, and never imported."""
    install_distribution(
        "icil-variant-benchmark",
        entry_points={"fake": "icil_variant_shadowed"},
        modules={"icil_variant_shadowed": variant("    pass")},
    )
    shadow = tmp_path / "checkout"
    shadow.mkdir()
    (shadow / "icil_variant_shadowed.py").write_text("raise SystemExit('shadow imported')\n")
    monkeypatch.syspath_prepend(str(shadow))
    spec = write_spec(
        fake_spec_doc(spec_doc, {**FAKE_PIN, "distribution": "icil-variant-benchmark"})
    )
    fake = discover(spec, load=True)["fake"]
    assert fake.benchmark is None and "icil_variant_shadowed" not in sys.modules
    (problem,) = fake.problems
    assert problem.startswith("icil_variant_shadowed resolves to ")
    assert str(shadow) in problem and "not a file of icil-variant-benchmark" in problem


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _install_with_record(site, name, module, code, wheel_url):
    info = site / f"{name.replace('-', '_')}-0.1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text(f"[icil.benchmarks]\nfake = {module}\n")
    (site / f"{module}.py").write_text(code)
    record = [f"{module}.py,{_record_hash(code.encode())},{len(code)}"]
    (info / "RECORD").write_text("\n".join([*record, f"{info.name}/RECORD,,"]) + "\n")
    (info / "direct_url.json").write_text(json.dumps({"url": wheel_url, "archive_info": {}}))
    return info, record


def test_installed_files_must_still_be_the_pinned_wheel(
    tmp_path, monkeypatch, spec_doc, write_spec
):
    """With a wheel pin, the bytes imported are held to the wheel: a file edited after install, or
    a wheel rebuilt in place without reinstalling, is refused."""
    name, module = "icil-variant-benchmark", "icil_variant_recorded"
    code = variant("    pass")
    wheel = tmp_path / "icil_variant_benchmark-0.1.0-py3-none-any.whl"
    site = tmp_path / "site"
    info, record = _install_with_record(site, name, module, code, wheel.as_uri())
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(f"{module}.py", code)
        zf.writestr(f"{info.name}/RECORD", "\n".join(record) + "\n")
    monkeypatch.syspath_prepend(str(site))
    pin = {**FAKE_PIN, "distribution": name, "wheel_sha256": plugins.sha256_file(wheel)}
    spec = write_spec(fake_spec_doc(spec_doc, pin))
    assert discover(spec, load=True)["fake"].ok
    sys.modules.pop(module, None)

    (site / f"{module}.py").write_text(code + "\n# edited after install\n")
    fake = discover(spec, load=True)["fake"]
    assert fake.benchmark is None and module not in sys.modules
    assert (
        f"{module}.py differs from its RECORD hash: the installed files are not the pinned wheel"
        in fake.problems
    )

    (site / f"{module}.py").write_text(code)
    with zipfile.ZipFile(wheel, "w") as zf:  # rebuilt with other code, never reinstalled
        zf.writestr(f"{module}.py", code + "# v2\n")
        zf.writestr(f"{info.name}/RECORD", f"{module}.py,{_record_hash(b'v2')},2\n")
    rebuilt = {**pin, "wheel_sha256": plugins.sha256_file(wheel)}
    fake = discover(write_spec(fake_spec_doc(spec_doc, rebuilt), name="rebuilt.json"), load=True)
    assert fake["fake"].benchmark is None
    assert any("cannot be confirmed" in p for p in fake["fake"].problems), fake["fake"].problems
