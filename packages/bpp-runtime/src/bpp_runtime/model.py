"""The network: built from the template, loaded from `model.safetensors`, checked before use.

**Building without a download.** The template keeps the vision encoder's `pretrained: true`, as
training had it: `TransformerObsEncoder` then leaves timm's weights alone (its own initializer
rejects the ViT's modules) and does not swap BatchNorm for GroupNorm. Only timm's download is
switched off, by building with `timm.create_model(..., pretrained=False)`. The ViT is the same
module either way - `pretrained_cfg`, and with it the image normalization, comes from the model
name - and every one of its tensors is overwritten by the strict load that follows. No network,
no Hugging Face cache.

**Loading.** The file is checked against the template first (header only), then read with
`safetensors`. Each tied alias is restored from its canonical tensor, and the result is loaded
with `strict=True`, so a missing or extra network tensor fails. Normalizer statistics are loaded
dynamically by the network and a strict load does not see them; their key set is the header
check's, and here they must also be finite with a non-zero scale. Every tensor must be finite.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import ARCHITECTURE
from .check import check
from .convert import dtype_name
from .template import NORMALIZER_PREFIX, load_config, load_tensors


class WeightsError(ValueError):
    """The weights cannot be loaded into the architecture."""


@contextlib.contextmanager
def _no_pretrained_download() -> Iterator[None]:
    """Within the block, `timm.create_model` builds its model without fetching weights."""
    import timm

    original = timm.create_model

    def create_model(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    timm.create_model = create_model
    try:
        yield
    finally:
        timm.create_model = original


def build_model(model_cfg: dict[str, Any]) -> Any:
    """The network `model_cfg` describes, randomly initialized, in eval mode, on the CPU.

    On the CPU, as the reference builds it: the diffusion scheduler is not a module, so its
    tables stay where they are computed, and computed on a GPU they could differ in the last bit.
    """
    import hydra

    with _no_pretrained_download():
        network = hydra.utils.instantiate(model_cfg)
    return network.eval()


def state_keys(network: Any) -> dict[str, tuple[int, ...]]:
    """Every tensor key of the network's state dict, with its shape."""
    import torch

    return {
        key: tuple(value.shape)
        for key, value in network.state_dict(keep_vars=True).items()
        if isinstance(value, torch.Tensor)
    }


def tied_tensors(network: Any) -> dict[str, str]:
    """`{alias: canonical}` for every state-dict key naming a tensor an earlier key names.

    By identity, not by storage: every zero-size `_dummy_variable` shares the null pointer
    without being tied to the others.
    """
    import torch

    first: dict[int, str] = {}
    tied: dict[str, str] = {}
    for key, value in network.state_dict(keep_vars=True).items():
        if not isinstance(value, torch.Tensor):
            continue
        canonical = first.setdefault(id(value), key)
        if canonical != key:
            tied[key] = canonical
    return tied


def validate_tensors(state: dict[str, Any]) -> None:
    """Every tensor finite; every normalizer scale non-zero. `WeightsError` otherwise."""
    import torch

    bad = []
    for key, value in state.items():
        if value.is_floating_point() and value.numel() and not bool(torch.isfinite(value).all()):
            bad.append(f"{key} holds non-finite values")
        if (
            key.startswith(NORMALIZER_PREFIX)
            and key.endswith(".scale")
            and bool((value == 0).any())
        ):
            bad.append(f"{key} holds a zero scale")
    if bad:
        raise WeightsError("; ".join(bad[:20]) + (f" ({len(bad)} in all)" if len(bad) > 20 else ""))


def load_policy(
    weights: str | os.PathLike[str],
    *,
    device: str = "cuda:0",
    template: str | os.PathLike[str] | None = None,
    architecture: str = ARCHITECTURE,
    weights_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """`(network, template config)`: the network with the weights at `weights` loaded, on
    `device`, in eval mode. `WeightsError` if the file is not the architecture's.
    """
    import torch
    from safetensors.torch import load_file

    report = check(
        weights, template=template, architecture=architecture, compute_sha256=bool(weights_sha256)
    )
    if not report.ok:
        raise WeightsError(f"{report.path}: " + "; ".join(report.errors))
    if weights_sha256 is not None and report.weights_sha256 != weights_sha256.lower():
        raise WeightsError(
            f"{report.path} has sha256 {report.weights_sha256}, expected {weights_sha256}"
        )
    config = load_config(template, architecture)

    state = load_file(str(Path(report.path)), device=str(device))
    # What was read, not only what the header said a moment ago: the normalizer takes any key.
    expected = load_tensors(template, architecture)
    if set(state) != set(expected) or any(
        list(state[k].shape) != expected[k]["shape"] or dtype_name(state[k]) != expected[k]["dtype"]
        for k in expected
    ):
        raise WeightsError(f"{report.path} changed after it was checked")
    validate_tensors(state)
    network = build_model(config["model"]).to(device)
    tied = config["tied_tensors"]
    for alias, canonical in tied.items():
        state[alias] = state[canonical]
    if tied_tensors(network) != tied:
        raise WeightsError("the network this code builds does not tie the template's tensors")
    result = network.load_state_dict(state, strict=True)
    if getattr(result, "missing_keys", None) or getattr(result, "unexpected_keys", None):
        raise WeightsError(f"strict load reported {result}")
    loaded = {NORMALIZER_PREFIX + key for key in network.normalizer.state_dict()}
    wanted = {key for key in state if key.startswith(NORMALIZER_PREFIX)}
    if loaded != wanted:
        raise WeightsError(f"the normalizer holds {sorted(loaded ^ wanted)[:10]} unexpectedly")
    # Eval mode, but requires_grad left as built, as the reference leaves it: a matmul on a
    # broadcast input (the attention pool's latent query) picks its kernel by requires_grad, and
    # a different kernel rounds differently. Inference runs under torch.inference_mode anyway.
    network.eval()
    del state, result
    if str(device).startswith("cuda"):
        # The file's tensors and the parameters were both on the device while loading: give the
        # copy back, so two policies of a duel sharing a GPU each hold one.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return network, config
