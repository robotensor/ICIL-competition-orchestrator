"""`icil-orchestrator benchmarks check <id>`: can this host run that benchmark for the spec?

The deploy check. It goes as far as it can without a simulator: the pin, the import, the ABI's
shape, `info`, the catalogue against the skills that name the benchmark, derivation (twice, since it
must be a pure function) and both command builders. It never runs a command - that needs the
simulator - and it reports every problem at once rather than stopping at the first.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .api import BENCHMARK_API_VERSION, PROMPT_FILE
from .plugins import SIMULATOR_MODULES, discover
from .units import DerivationError, check_unit

#: The environment variable name `check` hands `run_command`. Only a name: no key exists here.
CHECK_AUTHKEY_ENV = "ICIL_POLICY_AUTHKEY"
CHECK_ROOT = "/nonexistent/icil-orchestrator-check"


@dataclass
class CheckReport:
    name: str
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    info: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "problems": self.problems,
            "notes": self.notes,
            "info": self.info,
        }


def check_benchmark(spec: Any, name: str) -> CheckReport:
    report = CheckReport(name=name)
    plugged = discover(spec, load=(name,)).get(name)
    if plugged is None:
        report.problems.append("not declared in spec.json `benchmarks` and not installed")
        return report
    report.problems.extend(plugged.problems)
    report.notes.extend(plugged.notes)
    if not plugged.ok:
        return report

    benchmark = plugged.benchmark
    before = set(sys.modules)
    _check_info(report, benchmark)
    catalogue = _call(report, "catalogue", benchmark.catalogue)
    if catalogue is RAISED:
        catalogue = None
    elif not isinstance(catalogue, Mapping):
        report.problems.append(f"catalogue: returned {type(catalogue).__name__}, not a mapping")
        catalogue = None

    skills = [s for s in spec.all_skills if spec.benchmark_of(s) == name]
    if not skills:
        report.notes.append("no skill in spec.json runs on it")
    for skill in skills:
        _check_skill(report, spec, benchmark, skill, catalogue)

    leaked = sorted({m.split(".")[0] for m in set(sys.modules) - before} & SIMULATOR_MODULES)
    if leaked:
        report.problems.append(f"calling its pure methods imported {', '.join(leaked)}")
    return report


#: What `_call` returns for a method that raised (and was reported), so that a method that
#: *returned* None is still looked at - and refused - by its caller.
RAISED: Any = object()


def _call(report: CheckReport, what: str, fn: Any, **kwargs: Any) -> Any:
    try:
        return fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - reported, so the operator sees every problem
        report.problems.append(f"{what}: raised {type(exc).__name__}: {exc}")
        return RAISED


def _check_info(report: CheckReport, benchmark: Any) -> None:
    info = _call(report, "info", benchmark.info)
    if info is RAISED:
        return
    if not isinstance(info, Mapping):
        report.problems.append(f"info: returned {type(info).__name__}, not a mapping")
        return
    report.info = dict(info)
    if info.get("id") != report.name:
        report.problems.append(f"info: id is {info.get('id')!r}, not {report.name!r}")
    if info.get("api_version") != BENCHMARK_API_VERSION:
        report.problems.append(
            f"info: api_version is {info.get('api_version')!r}, not {BENCHMARK_API_VERSION}"
        )


def _check_skill(
    report: CheckReport,
    spec: Any,
    benchmark: Any,
    skill: str,
    catalogue: Mapping[str, Any] | None,
) -> None:
    suite, category = spec.suite(skill), spec.category(skill)
    if catalogue is not None:
        suites = catalogue.get("suites")
        if not isinstance(suites, Mapping) or suite not in suites:
            report.problems.append(f"{skill}: suite {suite!r} is not in its catalogue")
        categories = catalogue.get("categories")
        if category is not None and (
            not isinstance(categories, Mapping) or category not in categories
        ):
            report.problems.append(f"{skill}: category {category!r} is not in its catalogue")

    track = spec.track_of(skill)
    count = spec.units_per_skill(track)
    kwargs = dict(
        seed_material=f"icil-orchestrator benchmarks check|{skill}",
        count=count,
        suite=suite,
        category=category,
    )
    units = _call(report, f"{skill}: derive_units", benchmark.derive_units, **kwargs)
    if units is RAISED:
        return
    if not isinstance(units, (list, tuple)):
        report.problems.append(
            f"{skill}: derive_units: returned {type(units).__name__}, not a list"
        )
        return
    again = _call(report, f"{skill}: derive_units", benchmark.derive_units, **kwargs)
    units = list(units)
    if again is not RAISED and (not isinstance(again, (list, tuple)) or list(again) != units):
        report.problems.append(f"{skill}: derive_units is not a pure function of its arguments")
    if len(units) != count:
        report.problems.append(f"{skill}: derive_units returned {len(units)} units, not {count}")
        return
    try:
        for index, unit in enumerate(units):
            check_unit(skill, report.name, index, unit)
    except DerivationError as exc:
        report.problems.append(str(exc))
        return

    sample = {**units[0], "unit_id": f"{spec.skill_code(skill)}-000", "skill": skill}
    # The layout a duel uses: the prompt is materialized into its own directory, and each side runs
    # in another, so neither command's result.json can be taken for the other's.
    unit_dir = f"{CHECK_ROOT}/{sample['unit_id']}"
    prompt_dir = f"{unit_dir}/prompt"
    _check_argv(
        report,
        f"{skill}: materialize_command",
        benchmark.materialize_command,
        unit=sample,
        out_dir=prompt_dir,
    )
    _check_argv(
        report,
        f"{skill}: run_command",
        benchmark.run_command,
        unit=sample,
        prompt=f"{prompt_dir}/{PROMPT_FILE}",
        out_dir=f"{unit_dir}/challenger",
        policy_address=f"{CHECK_ROOT}/policy.sock",
        authkey_env=CHECK_AUTHKEY_ENV,
    )


def _check_argv(report: CheckReport, what: str, fn: Any, **kwargs: Any) -> None:
    argv = _call(report, what, fn, **kwargs)
    if argv is RAISED:
        return
    if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(a, str) for a in argv):
        report.problems.append(f"{what}: expected a non-empty list of strings, got {argv!r}")
