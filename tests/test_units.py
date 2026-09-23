"""Deriving a duel's units from a benchmark in another distribution.

Only the benchmark knows what one of its units is. What the orchestrator keeps is the identity of a
unit and the seed material it is derived from, both reproducible from the published record.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import FAKE_SITE
from vector_orchestrator.benchmarks import units as plugin_units
from vector_orchestrator.benchmarks.units import DerivationError
from vector_orchestrator.ids import SubmissionRef, duel_id

CHALLENGER = SubmissionRef.make("org/challenger", "1" * 40)
KING = SubmissionRef.make("org/king", "2" * 40)


class Recording:
    """Wraps a plugin, recording what it was asked."""

    def __init__(self, inner, **mutations):
        self.inner = inner
        self.calls: list[dict] = []
        self.mutations = mutations

    def derive_units(self, **kw):
        self.calls.append(kw)
        if self.mutations.get("raises"):
            raise RuntimeError("catalogue unreachable")
        units = self.inner.derive_units(**kw)
        if "count" in self.mutations:
            units = units[: self.mutations["count"]]
        for unit in units:
            unit.update(self.mutations.get("update", {}))
            for key in self.mutations.get("drop", ()):
                unit.pop(key, None)
        return units


@pytest.fixture
def fake(fake_installed):
    import vector_fake_benchmark

    return vector_fake_benchmark.BENCHMARK


def derive(spec, plugin, did, size="light"):
    return plugin_units.plugin_units(spec, "franka_1arm", did, size, resolve=lambda name: plugin)


def test_units_come_from_the_benchmark_and_ids_from_the_competition(fake_spec, fake):
    plugin = Recording(fake, update={"unit_id": "mine", "seed": 1})
    did = duel_id(fake_spec.version, "franka_1arm", CHALLENGER, KING)
    units = derive(fake_spec, plugin, did)

    assert [u["unit_id"] for u in units] == [
        "fp-000",
        "fp-001",
        "fp-002",
        "fs-003",
        "fs-004",
        "fs-005",
        "fu-006",
        "fu-007",
        "fu-008",
    ]
    assert [c["seed_material"] for c in plugin.calls] == [
        f"{did}|franka_pick_and_place",
        f"{did}|franka_stacking",
        f"{did}|franka_press_push",
    ]
    assert [(c["suite"], c["category"], c["count"]) for c in plugin.calls] == [
        ("franka_1arm", "pick_and_place", 3),
        ("franka_1arm", "stacking", 3),
        ("franka_1arm", "press_push", 3),
    ]
    # A benchmark cannot choose a unit's identity or its seed.
    assert "mine" not in {u["unit_id"] for u in units} and 1 not in {u["seed"] for u in units}
    for unit in units:
        assert (
            unit["benchmark"] == "fake"
            and unit["instance"] == 0
            and unit["demo"] == unit["unit_id"]
        )
        assert {"scene_seed", "embodiment"} <= set(unit["instance_params"])
        assert unit["category"] == fake_spec.category(unit["skill"])


def test_derivation_is_reproducible_from_the_published_record(fake_spec, fake):
    did = duel_id(fake_spec.version, "franka_1arm", CHALLENGER, KING)
    other = duel_id(fake_spec.version, "franka_1arm", CHALLENGER, None)
    assert derive(fake_spec, fake, did) == derive(fake_spec, fake, did)
    assert [u["seed"] for u in derive(fake_spec, fake, did)] != [
        u["seed"] for u in derive(fake_spec, fake, other)
    ]


#: Frozen units for one duel. The same values must come out on every supported Python version
#: (CI runs 3.10 and 3.12), because a third party re-derives them from the record.
GOLDEN = [
    ("fp-000", 3575735063),
    ("fp-001", 2983153288),
    ("fp-002", 543356373),
    ("fs-003", 1670043122),
    ("fs-004", 1208218009),
    ("fs-005", 3109376669),
    ("fu-006", 3672912370),
    ("fu-007", 3552488969),
    ("fu-008", 3439904202),
]


def test_unit_ids_and_seeds_are_frozen_for_a_duel(fake_spec, fake):
    did = duel_id(fake_spec.version, "franka_1arm", CHALLENGER, KING)
    units = derive(fake_spec, fake, did)
    assert did == "7eab00ae0613e3785d960a714c09c17cfbdd39c8c0d60b06b707b833c8b57c46"
    assert [(u["unit_id"], u["seed"]) for u in units] == GOLDEN


def test_the_same_units_come_out_of_a_fresh_interpreter(fake_spec, fake):
    """A fresh process has another string-hash salt, so a derivation leaning on `hash()` or on
    process state would differ here. Across Python versions the golden values above decide."""
    code = f"""
import json, sys
sys.path.insert(0, {str(FAKE_SITE)!r})
import vector_fake_benchmark
from vector_orchestrator.benchmarks.units import plugin_units
from vector_orchestrator.ids import SubmissionRef, duel_id
from vector_orchestrator.spec import load_spec_file
spec = load_spec_file({str(fake_spec.path)!r})
c = SubmissionRef.make("org/challenger", "1" * 40)
k = SubmissionRef.make("org/king", "2" * 40)
did = duel_id(spec.version, "franka_1arm", c, k)
units = plugin_units(spec, "franka_1arm", did, "light", resolve=lambda n: vector_fake_benchmark.BENCHMARK)
print(json.dumps(units, sort_keys=True))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    did = duel_id(fake_spec.version, "franka_1arm", CHALLENGER, KING)
    here = derive(fake_spec, fake, did)
    assert json.loads(out.stdout) == json.loads(json.dumps(here, sort_keys=True))


@pytest.mark.parametrize(
    "mutations, message",
    [
        ({"count": 1}, "franka_pick_and_place: fake returned 1 units, not the 3 franka_1arm/light"),
        ({"raises": True}, "fake could not derive its units: catalogue unreachable"),
        ({"drop": ("task_label",)}, "fake unit 0 has no task_label"),
        ({"drop": ("instance_params",)}, "fake unit 0 has no instance_params mapping"),
        ({"update": {"instance_params": {"embodiment": []}}}, "instance_params lacks scene_seed"),
    ],
)
def test_a_benchmark_returning_the_wrong_units_is_named(fake_spec, fake, mutations, message):
    with pytest.raises(DerivationError, match=message.replace("/", ".")):
        derive(fake_spec, Recording(fake, **mutations), "d" * 64)


def test_a_unit_that_is_not_a_mapping_is_refused(fake_spec):
    class Bad:
        def derive_units(self, **kw):
            return ["not a unit"] * kw["count"]

    with pytest.raises(DerivationError, match="unit 0 is str, not a mapping"):
        derive(fake_spec, Bad(), "d" * 64)


def test_by_default_the_pinned_installed_plugin_is_used(fake_installed, fake_spec):
    units = plugin_units.plugin_units(fake_spec, "franka_1arm", "d" * 64, "smoke")
    assert len(units) == 3
