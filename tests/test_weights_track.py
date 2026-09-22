"""The BPP subnet lane: a weights-only track, chain-seeded units and a paired crown rule."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from icil_orchestrator.benchmarks.units import plugin_units, seed_key, seed_material
from icil_orchestrator.duel import score
from icil_orchestrator.duel.orchestrate import DuelRequest
from icil_orchestrator.duel.runtime import SubmissionRefused
from icil_orchestrator.duel.weights_runtime import WeightsCheck, WeightsPolicyRuntime
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.spec import load_spec_file, validate_spec

ROOT = Path(__file__).resolve().parents[1]
BPP_SPEC = ROOT / "specs" / "bpp_l1.json"
SHA = "a" * 40
HASH = "0x" + "b" * 64


@pytest.fixture
def bpp_doc():
    return json.loads(BPP_SPEC.read_text())


@pytest.fixture
def bpp_spec():
    return load_spec_file(BPP_SPEC)


# -- the contract ----------------------------------------------------------------------------


def test_the_bpp_contract_is_a_valid_weights_track(bpp_spec):
    assert bpp_spec.tracks == ("bpp_l1",)
    assert bpp_spec.submission_kind == "weights"
    assert bpp_spec.model["weights_file"] == "model.safetensors"
    assert bpp_spec.protocol("bpp_l1") == "different_initial_state"
    assert bpp_spec.paired_alpha("bpp_l1") == 0.05
    # 10 units per task on the 16-task suite, the launch size.
    assert bpp_spec.units_per_side("bpp_l1") == 160
    assert bpp_spec.baseline("bpp_l1")["size"] == "smoke"


def test_a_code_spec_still_needs_its_sandbox_and_has_no_model(spec):
    assert spec.submission_kind == "code" and spec.paired_alpha("franka_1arm") is None
    with pytest.raises(KeyError):
        _ = spec.model


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda d: d["submission"].pop("model"), "submission.model mapping"),
        (
            lambda d: d["submission"]["model"].update(policy="not a class"),
            "submission.model.policy",
        ),
        (
            lambda d: d["submission"]["model"].update(allowed_files=["README.md"]),
            "submission.model.allowed_files",
        ),
        (lambda d: d["submission"].update(kind="docker"), "submission.kind"),
        (lambda d: d["duel"]["crown"].update(alpha=1.5), "duel.crown.alpha"),
        (lambda d: d["duel"]["crown"].update(paired_test="t"), "duel.crown.paired_test"),
        (lambda d: d["baselines"]["bpp_l1"].update(size="huge"), "baselines.bpp_l1.size"),
    ],
)
def test_the_weights_contract_refuses_what_it_cannot_honour(bpp_doc, edit, problem):
    doc = copy.deepcopy(bpp_doc)
    edit(doc)
    assert any(error.startswith(problem) for error in validate_spec(doc)), validate_spec(doc)


def test_a_weights_spec_needs_no_sandbox(bpp_doc):
    assert "sandbox" not in bpp_doc["submission"] and validate_spec(bpp_doc) == []


# -- chain entropy ---------------------------------------------------------------------------


def request(**seed) -> DuelRequest:
    return DuelRequest(
        "bpp_l1",
        SubmissionRef.make("m/challenger", SHA),
        SubmissionRef.make("m/king", "c" * 40),
        "smoke",
        block=3,
        **seed,
    )


def test_a_request_carries_its_seed_block_through_request_json(bpp_spec):
    req = request(seed_block=123, seed_block_hash=HASH)
    doc = req.as_dict(bpp_spec)
    assert doc["seed_block"] == 123 and doc["seed_block_hash"] == HASH
    assert DuelRequest.from_dict(doc) == req
    assert req.entropy == f"123:{HASH}"


def test_a_request_without_a_seed_block_is_written_as_before(bpp_spec):
    doc = request().as_dict(bpp_spec)
    assert "seed_block" not in doc and request().entropy is None
    assert DuelRequest.from_dict(doc) == request()


@pytest.mark.parametrize(
    "seed",
    [
        {"seed_block": 1},
        {"seed_block_hash": HASH},
        {"seed_block": 1, "seed_block_hash": "0x1234"},
        {"seed_block": -1, "seed_block_hash": HASH},
    ],
)
def test_a_half_or_malformed_seed_is_refused(seed):
    with pytest.raises(ValueError):
        request(**seed)


def test_entropy_moves_every_unit_and_the_same_entropy_derives_the_same_ones(spec):
    calls = []

    class Recorder:
        def derive_units(self, *, seed_material, count, suite, category):
            calls.append(seed_material)
            return [
                {
                    "task": "t",
                    "task_label": "T",
                    "instance_params": {"scene_seed": None, "embodiment": "e"},
                }
                for _ in range(count)
            ]

    plain = plugin_units(spec, "franka_1arm", "d", "smoke", resolve=lambda _: Recorder())
    seeded = plugin_units(
        spec, "franka_1arm", "d", "smoke", resolve=lambda _: Recorder(), entropy="7:0xab"
    )
    assert calls[0] == seed_material("d", calls[0].split("|")[-1])
    assert calls[-1].startswith("d|7:0xab|")
    assert [u["seed"] for u in plain] != [u["seed"] for u in seeded]
    again = plugin_units(
        spec, "franka_1arm", "d", "smoke", resolve=lambda _: Recorder(), entropy="7:0xab"
    )
    assert again == seeded
    assert seed_key("d") == "d" and seed_key("d", "7:0xab") == "d|7:0xab"


# -- the crown rule --------------------------------------------------------------------------


def test_the_sign_test():
    assert score.sign_test_p(0, 0) == 1.0
    assert score.sign_test_p(5, 0) == pytest.approx(1 / 32)
    assert score.sign_test_p(3, 3) == pytest.approx(42 / 64)


def units(pairs):
    """Rows of (challenger_success, king_success), all in one skill."""
    return [
        {"skill": "s", "void": False, "challenger_success": c, "king_success": k} for c, k in pairs
    ]


def test_a_met_margin_on_a_few_units_is_not_significant():
    # 2 of 3 for the challenger against 1 of 3: +33 points, but one discordant unit.
    rows = units([(True, True), (True, False), (False, False)])
    v = score.verdict(rows, 3.0, ["s"], paired_alpha=0.05)
    assert not v.dethroned and v.reason == "not-significant"
    assert v.as_dict()["paired_p_value"] == pytest.approx(0.5)
    # Without a paired test the margin alone decides, as before.
    assert score.verdict(rows, 3.0, ["s"]).dethroned


def test_a_clear_win_moves_the_crown():
    rows = units([(True, False)] * 8 + [(True, True)] * 2 + [(False, False)] * 6)
    v = score.verdict(rows, 3.0, ["s"], paired_alpha=0.05)
    assert v.dethroned and v.reason == "margin-met"
    assert v.paired_p_value == pytest.approx(1 / 256)


# -- the weights runtime ---------------------------------------------------------------------


def hub(files: dict[str, int]):
    info = SimpleNamespace(
        sha=SHA, siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in files.items()]
    )
    return SimpleNamespace(repo_info=lambda *a, **k: info)


def runtime(tmp_path, bpp_spec, files, checker=None, downloads=None):
    def download(repo, *, revision, repo_type, local_dir):
        (downloads if downloads is not None else []).append(repo)
        Path(local_dir, "model.safetensors").write_bytes(b"weights")

    return WeightsPolicyRuntime(
        bpp_spec,
        python="python",
        cache_root=tmp_path / "cache",
        api=hub(files),
        download=download,
        checker=checker or (lambda path: WeightsCheck(ok=True, weights_sha256="f" * 64)),
        hello=False,
    )


def test_a_repository_holding_anything_but_weights_and_a_readme_is_refused(tmp_path, bpp_spec):
    rt = runtime(tmp_path, bpp_spec, {"model.safetensors": 7, "policy.py": 10})
    with pytest.raises(SubmissionRefused) as refused:
        rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")
    assert refused.value.step == "fetch" and "policy.py" in refused.value.reason


def test_a_repository_without_its_weights_file_is_refused(tmp_path, bpp_spec):
    rt = runtime(tmp_path, bpp_spec, {"README.md": 5})
    with pytest.raises(SubmissionRefused, match="holds no model.safetensors"):
        rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")


def test_only_the_weights_file_is_fetched_and_the_manifest_is_the_validators(tmp_path, bpp_spec):
    downloads: list[str] = []
    rt = runtime(tmp_path, bpp_spec, {"model.safetensors": 7, "README.md": 5}, downloads=downloads)
    fetched = rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")
    assert downloads == ["m/r"] and sorted(p.name for p in fetched.root.iterdir()) == [
        "model.safetensors"
    ]
    prepared = rt.prepare(fetched, workdir=tmp_path / "w" / "check")
    manifest = json.loads(Path(prepared.handle).read_text())
    assert manifest["policy"] == "bpp_runtime.policy:BPPPolicy"
    assert manifest["kwargs"]["weights"] == str((fetched.root / "model.safetensors").resolve())
    assert prepared.image == "sha256:" + "f" * 64 and prepared.action_type == "ee"


def test_weights_that_fail_the_template_check_are_refused(tmp_path, bpp_spec):
    rt = runtime(
        tmp_path,
        bpp_spec,
        {"model.safetensors": 7},
        checker=lambda path: WeightsCheck(ok=False, errors=("key x: wrong shape",)),
    )
    fetched = rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")
    with pytest.raises(SubmissionRefused) as refused:
        rt.prepare(fetched, workdir=tmp_path / "w" / "check")
    assert refused.value.step == "check" and "wrong shape" in refused.value.reason
