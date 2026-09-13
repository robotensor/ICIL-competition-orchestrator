"""The benchmark plugin ABI, exercised through a minimal in-test plugin.

Nothing here imports a benchmark: the point of the ABI is that this repository can describe and
check one without having it installed.
"""

from __future__ import annotations

from typing import Any

from icil_orchestrator.benchmarks import (
    BENCHMARK_API_VERSION,
    COMMAND_METHODS,
    PURE_METHODS,
    Benchmark,
    validate_plugin,
)
from icil_orchestrator.benchmarks.api import METHODS


class Minimal:
    """A minimal plugin that satisfies the ABI, and the base for every mutilation below."""

    id = "minimal"
    api_version = BENCHMARK_API_VERSION

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "api_version": self.api_version}

    def catalogue(self) -> dict[str, Any]:
        return {"suites": {"s": ["t0", "t1"]}, "categories": {"c": "C"}, "tasks": {}}

    def derive_units(
        self, *, seed_material: str, count: int, suite: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        return [{"task": f"t{i % 2}", "suite": suite} for i in range(count)]

    def verify_prompt(self, *, path: str, unit: Any) -> dict[str, Any]:
        return {"ok": True, "sha256": "0" * 64, "problems": []}

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        return {"success": True, "void": False, "steps": 1, "error": None}

    def materialize_command(self, *, unit: Any, out_dir: str) -> list[str]:
        return ["bench", "materialize", "--out", out_dir]

    def run_command(
        self,
        *,
        unit: Any,
        prompt: str,
        out_dir: str,
        policy_address: str,
        authkey_env: str,
        **extra: Any,
    ) -> list[str]:
        return ["bench", "run", prompt, policy_address, "--authkey-env", authkey_env]


def test_a_conforming_plugin_validates():
    assert validate_plugin(Minimal()) == []
    assert isinstance(Minimal(), Benchmark)


def test_the_two_halves_together_are_the_whole_surface():
    assert set(PURE_METHODS) | set(COMMAND_METHODS) == set(METHODS)
    assert not set(PURE_METHODS) & set(COMMAND_METHODS)


def test_every_missing_method_is_reported_by_name():
    for name in METHODS:
        broken = type("Broken", (Minimal,), {name: None})()
        errors = validate_plugin(broken)
        assert f"{name}: missing" in errors, (name, errors)


def test_a_plugin_without_run_command_is_refused_with_the_reason():
    broken = type("Broken", (Minimal,), {"run_command": None})()
    assert validate_plugin(broken) == ["run_command: missing"]


def test_a_method_that_is_not_callable_is_refused():
    broken = type("Broken", (Minimal,), {"catalogue": 3})()
    assert "catalogue: not callable" in validate_plugin(broken)


def test_run_command_must_take_the_authkey_env_name():
    """The key travels by environment variable name; a builder that cannot be told the name would
    have to put the key itself on the command line."""

    class Broken(Minimal):
        def run_command(self, *, unit, prompt, out_dir, policy_address):  # type: ignore[override]
            return []

    assert validate_plugin(Broken()) == ["run_command: does not accept authkey_env"]


def test_derive_units_must_take_a_category():
    class Broken(Minimal):
        def derive_units(self, *, seed_material, count, suite):  # type: ignore[override]
            return []

    assert validate_plugin(Broken()) == ["derive_units: does not accept category"]


def test_a_method_taking_kwargs_is_accepted():
    """A plugin is free to absorb the call; only refusing a keyword is an error."""

    class Loose(Minimal):
        def run_command(self, **kw: Any) -> list[str]:  # type: ignore[override]
            return []

    assert validate_plugin(Loose()) == []


def test_an_id_must_be_a_non_empty_string():
    for value in ("", None, 7):
        broken = type("Broken", (Minimal,), {"id": value})()
        assert "id: expected a non-empty string" in validate_plugin(broken)


def test_a_plugin_speaking_another_abi_version_is_refused_with_both_numbers():
    other = BENCHMARK_API_VERSION + 1
    broken = type("Broken", (Minimal,), {"api_version": other})()
    assert validate_plugin(broken) == [
        f"api_version: speaks {other}, this orchestrator speaks {BENCHMARK_API_VERSION}"
    ]


def test_a_boolean_is_not_an_api_version():
    """`True == 1` in Python, so a plugin with `api_version = True` would otherwise pass."""
    broken = type("Broken", (Minimal,), {"api_version": True})()
    assert "api_version: expected an integer" in validate_plugin(broken)


def test_errors_accumulate_rather_than_stopping_at_the_first():
    broken = type("Broken", (Minimal,), {"id": "", "api_version": "1", "info": None})()
    assert len(validate_plugin(broken)) == 3
