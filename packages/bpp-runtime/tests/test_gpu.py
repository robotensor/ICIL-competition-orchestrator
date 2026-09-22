"""Convert, load, serve and parity on the real BRL1 checkpoints. Needs torch, CUDA, the vendored
behavior_prompting and the checkpoints (`BPP_CHECKPOINTS`, default the BRL1 cache); `BPP_PROMPT`
names a benchmark prompt.npz for parity, else a synthetic demonstration is used."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("behavior_prompting")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from bpp_runtime import WEIGHTS_FILENAME, parity  # noqa: E402
from bpp_runtime.check import check  # noqa: E402
from bpp_runtime.convert import convert  # noqa: E402
from bpp_runtime.model import WeightsError, load_policy  # noqa: E402
from bpp_runtime.policy import BPPPolicy  # noqa: E402

CHECKPOINTS = Path(os.environ.get("BPP_CHECKPOINTS", "/root/robotensor/.cache/brl1/checkpoints"))
BASE = CHECKPOINTS / "epoch0000.ckpt"
if not BASE.exists():
    pytest.skip(f"no checkpoint at {BASE}", allow_module_level=True)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """epoch0000 converted: `(directory, check report)`."""
    out = tmp_path_factory.mktemp("converted")
    report, info = convert(BASE, out)
    assert info["dropped_non_tensors"] == ["_extra_training_split_info"]
    assert info["config_differences"] == []
    return out, report


@pytest.fixture(scope="module")
def policy(converted):
    return BPPPolicy(weights=str(converted[0] / WEIGHTS_FILENAME))


@pytest.fixture(scope="module")
def prompt_file(tmp_path_factory):
    if os.environ.get("BPP_PROMPT"):
        return Path(os.environ["BPP_PROMPT"])
    return synthetic_prompt(tmp_path_factory.mktemp("prompt") / "prompt.npz")


def synthetic_prompt(path: Path, count: int = 40) -> Path:
    """A smooth two-arm demonstration of noise images, written as a benchmark prompt file."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, count)
    endpose = np.zeros((count, 16))
    for start, x in ((0, -0.4), (8, 0.4)):
        endpose[:, start] = x + 0.05 * np.sin(3 * t)
        endpose[:, start + 1] = -0.2 + 0.05 * t
        endpose[:, start + 2] = 0.98 - 0.1 * t
        half = 0.3 * t
        endpose[:, start + 3] = np.cos(half)
        endpose[:, start + 6] = np.sin(half)
        endpose[:, start + 7] = 1.0 - t
    arrays = {
        f"frames_{camera}": rng.integers(0, 256, (count, 240, 320, 3), dtype=np.uint8)
        for camera in ("head_camera", "left_camera", "right_camera")
    }
    np.savez_compressed(
        path,
        **arrays,
        endpose=endpose,
        qpos=np.zeros((count, 16)),
        actions=np.zeros((count - 1, 16)),
        times=t,
        frequency=np.asarray(15.0),
        meta=json.dumps({"privileged": True}),
    )
    return path


def observation(arrays, row):
    return {
        k: v[row] for k, v in arrays.items() if k.startswith("frames_") or k in ("qpos", "endpose")
    }


def test_conversion_passes_the_check_and_is_deterministic(converted, tmp_path):
    directory, report = converted
    assert report.ok, report.errors
    assert report.tensor_count == 744
    assert report.param_count == 541_023_680
    again, _ = convert(BASE, tmp_path)
    assert again.weights_sha256 == report.weights_sha256


def test_the_policy_follows_the_protocol(policy, prompt_file):
    arrays, info = parity.read_prompt(prompt_file)
    with pytest.raises(RuntimeError, match="before reset"):
        policy.act(observation(arrays, 0))
    policy.reset(0)
    with pytest.raises(RuntimeError, match="before set_demonstration"):
        policy.act(observation(arrays, 0))

    def episode(seed, steps=14):
        policy.reset(seed)
        policy.set_demonstration(arrays, info)
        return np.stack([policy.act(observation(arrays, i))["action"] for i in range(steps)])

    first = episode(0)
    assert first.shape == (14, 16) and first.dtype == np.float64
    assert np.all(np.isfinite(first))
    assert np.all((first[:, [7, 15]] >= 0) & (first[:, [7, 15]] <= 1))
    assert policy.predictions >= 2  # 14 steps at 12 executed actions per prediction
    np.testing.assert_array_equal(episode(0), first)  # the seed fixes the noise
    assert not np.array_equal(episode(1), first)

    policy.set_demonstration(arrays, info)
    policy.reset(0)  # reset forgets the prompt
    with pytest.raises(RuntimeError, match="before set_demonstration"):
        policy.act(observation(arrays, 0))
    policy.close()  # the network stays loaded
    assert next(policy.network.parameters()).device.type == "cuda"


def test_the_converted_weights_act_exactly_as_the_checkpoint(converted, prompt_file):
    report = parity.run(BASE, converted[0], prompt_file, seed=3, steps=13)
    assert report["predictions"] == 2
    assert report["exact"], report


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("normalizer.params_dict.action.scale", 0.0, "zero scale"),
        ("normalizer.prompt_normalizer.params_dict.action.offset", float("nan"), "non-finite"),
        ("model.final_conv.1.bias", float("inf"), "non-finite"),
    ],
)
def test_the_loader_refuses_bad_values(converted, tmp_path, key, value, message):
    from safetensors.torch import load_file, save_file

    state = load_file(str(converted[0] / WEIGHTS_FILENAME))
    state[key] = state[key].clone()
    state[key][0] = value
    bad = tmp_path / WEIGHTS_FILENAME
    save_file(state, str(bad), metadata={"format": "pt"})
    assert check(bad).ok  # a header cannot show a value
    with pytest.raises(WeightsError, match=message):
        load_policy(bad, device="cuda:0")


def test_the_loader_refuses_a_file_that_fails_the_check(converted, tmp_path):
    from safetensors.torch import load_file, save_file

    state = load_file(str(converted[0] / WEIGHTS_FILENAME))
    del state["normalizer.prompt_normalizer.params_dict.action.scale"]
    bad = tmp_path / WEIGHTS_FILENAME
    save_file(state, str(bad), metadata={"format": "pt"})
    with pytest.raises(WeightsError, match="is missing"):
        load_policy(bad, device="cuda:0")
    with pytest.raises(WeightsError, match="sha256"):
        load_policy(converted[0], device="cuda:0", weights_sha256="0" * 64)
