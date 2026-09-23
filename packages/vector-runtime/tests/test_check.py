"""`vector-runtime check`: the template's tensors exactly, a sane file, and its sha256. No torch."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from st_testing import TENSORS, safetensors_bytes, write_template
from vector_runtime import WEIGHTS_FILENAME
from vector_runtime.check import check, check_weights
from vector_runtime.cli import main
from vector_runtime.template import load_config, load_tensors


@pytest.fixture
def template(tmp_path):
    return write_template(tmp_path / "arch")


def weights(tmp_path, data: bytes, name: str = WEIGHTS_FILENAME):
    directory = tmp_path / "submission"
    directory.mkdir(exist_ok=True)
    (directory / name).write_bytes(data)
    return directory


def test_a_matching_file_passes_with_its_sha256(tmp_path, template):
    data = safetensors_bytes(metadata={"format": "pt"})
    report = check(weights(tmp_path, data), template=template)
    assert report.ok, report.errors
    assert report.weights_sha256 == hashlib.sha256(data).hexdigest()
    assert report.tensor_count == len(TENSORS)
    assert report.param_count == 36
    assert report.file_bytes == len(data)
    assert report.metadata == {"format": "pt"}
    assert report.to_dict()["ok"] is True


def test_a_file_path_works_as_well_as_its_directory(tmp_path, template):
    directory = weights(tmp_path, safetensors_bytes())
    assert check(directory / WEIGHTS_FILENAME, template=template).ok


def test_a_missing_tensor_fails(tmp_path, template):
    tensors = {k: v for k, v in TENSORS.items() if k != "normalizer.params_dict.action.scale"}
    report = check(weights(tmp_path, safetensors_bytes(tensors)), template=template)
    assert report.errors == ["tensor normalizer.params_dict.action.scale is missing"]


def test_an_extra_tensor_fails(tmp_path, template):
    # a normalizer key the network would load silently: the check is the only thing that sees it
    tensors = {**TENSORS, "normalizer.params_dict.evil.scale": ("F32", [3])}
    report = check(weights(tmp_path, safetensors_bytes(tensors)), template=template)
    assert report.errors == [
        "tensor normalizer.params_dict.evil.scale is not part of the architecture"
    ]


def test_a_wrong_shape_fails(tmp_path, template):
    tensors = {**TENSORS, "model.weight": ("F32", [3, 4])}
    report = check(weights(tmp_path, safetensors_bytes(tensors)), template=template)
    assert report.errors == ["tensor model.weight: shape [3, 4] != [4, 3]"]


def test_a_wrong_dtype_fails(tmp_path, template):
    tensors = {**TENSORS, "model.bias": ("BF16", [4])}
    report = check(weights(tmp_path, safetensors_bytes(tensors)), template=template)
    assert report.errors == ["tensor model.bias: dtype BF16 != F32"]


def test_a_huge_header_fails_without_being_read(tmp_path, template):
    report = check(weights(tmp_path, safetensors_bytes(length=1 << 40)), template=template)
    assert not report.ok
    assert "at most" in report.errors[0]


def test_a_header_over_the_cap_fails(tmp_path, template):
    report = check(weights(tmp_path, safetensors_bytes()), template=template, max_header_bytes=64)
    assert not report.ok
    assert "at most 64" in report.errors[0]


def test_an_oversized_file_fails_before_its_header_is_read(tmp_path, template):
    data = safetensors_bytes()
    report = check(weights(tmp_path, data), template=template, max_file_bytes=len(data) - 1)
    assert report.errors == [f"the file is {len(data)} bytes; at most {len(data) - 1}"]
    assert report.weights_sha256 is None


def test_garbage_fails(tmp_path, template):
    report = check(weights(tmp_path, b"\x80\x04\x95 not safetensors at all"), template=template)
    assert not report.ok
    assert report.errors[0].startswith("not a valid safetensors file")


def test_appended_bytes_fail(tmp_path, template):
    report = check(weights(tmp_path, safetensors_bytes(extra=b"PK\x03\x04")), template=template)
    assert not report.ok


def test_a_directory_without_the_file_fails(tmp_path, template):
    directory = weights(tmp_path, safetensors_bytes(), name="pytorch_model.bin")
    report = check(directory, template=template)
    assert not report.ok
    assert report.other_files == ["pytorch_model.bin"]


def test_other_files_are_a_warning(tmp_path, template):
    directory = weights(tmp_path, safetensors_bytes())
    (directory / "README.md").write_text("hello")
    report = check(directory, template=template)
    assert report.ok
    assert report.other_files == ["README.md"]
    assert report.warnings


def test_a_symbolic_link_is_followed(tmp_path, template):
    # a Hugging Face snapshot directory holds links into its blob store
    blob = tmp_path / "blob"
    blob.write_bytes(safetensors_bytes())
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / WEIGHTS_FILENAME).symlink_to(blob)
    assert check(directory, template=template).ok


def test_the_template_directory_can_come_from_the_environment(tmp_path, template, monkeypatch):
    monkeypatch.setenv("VECTOR_RUNTIME_TEMPLATE", str(template))
    assert check_weights(weights(tmp_path, safetensors_bytes())).ok
    monkeypatch.setenv("VECTOR_RUNTIME_TEMPLATE", str(tmp_path / "nowhere"))
    assert "cannot read the template" in check_weights(weights(tmp_path, b"")).errors[0]


def test_the_cli_prints_a_report_and_exits_by_it(tmp_path, template, capsys):
    good = weights(tmp_path, safetensors_bytes())
    assert main(["check", "--weights", str(good), "--template", str(template)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    (good / WEIGHTS_FILENAME).write_bytes(safetensors_bytes({"model.bias": ("F32", [4])}))
    assert main(["check", "--weights", str(good), "--template", str(template)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_the_packaged_template_is_the_bpp_architecture():
    tensors = load_tensors()
    config = load_config()
    assert len(tensors) == 744
    assert sum(_numel(t["shape"]) for t in tensors.values()) == 541_023_680
    assert {t["dtype"] for t in tensors.values()} == {"F32"}
    assert not set(config["tied_tensors"]) & set(tensors)
    assert set(config["tied_tensors"].values()) <= set(tensors)
    assert config["exec_action_horizon"] == 12
    assert config["action_horizon"] == 16
    assert config["model"]["obs_encoder"]["obs_encoder"]["obs_encoder"]["pretrained"] is True


def test_the_check_imports_no_torch_or_numpy():
    code = (
        "import sys, vector_runtime.check, vector_runtime.cli; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'numpy', 'safetensors')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n
