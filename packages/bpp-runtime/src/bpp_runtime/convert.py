"""A BPP training checkpoint -> `model.safetensors`, the one file a submission holds.

    bpp-runtime convert --ckpt epoch=0004.ckpt --out DIR

What a miner runs, on a checkpoint of their own. A checkpoint is a dill pickle of the training
workspace (`cfg`, `state_dicts.{model,optimizer}`, `pickles`): unpickling one runs whatever it
says, so this module is the only code in the runtime that unpickles, and it must only ever be
handed a file its caller made. The validator never unpickles anything a miner sends.

What is written is `state_dicts.model`, the EMA weights the workspace saves there:

- only tensors: `BasePolicy.state_dict()` adds `_extra_training_split_info`, which is dropped;
- one key per tensor: the template's tied aliases (the shared prompt/receding encoder, the shared
  wrist-camera ViT) are checked to be equal to their canonical tensor and left out, since
  safetensors refuses shared tensors and the loader restores them;
- zero-size `_dummy_variable`s are kept: they are part of the network's state dict.

The result is then checked against the template exactly as a validator will check it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import ARCHITECTURE, WEIGHTS_FILENAME
from .check import CheckReport, check, sha256_file
from .template import load_config, load_tensors

__all__ = ["convert", "load_checkpoint", "sha256_file"]

#: The non-tensor entry `BasePolicy.state_dict()` adds.
SPLIT_INFO_KEY = "_extra_training_split_info"
#: The header metadata written. The same checkpoint always converts to the same bytes, so the
#: same weights always have the same `weights_sha256`.
METADATA = {"format": "pt"}


class ConvertError(ValueError):
    """The checkpoint cannot be expressed as this architecture's weights."""


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Unpickle a training checkpoint onto the CPU. Only ever a file the caller owns."""
    import dill
    import torch

    with open(path, "rb") as handle:
        payload = torch.load(handle, map_location="cpu", pickle_module=dill, weights_only=False)
    if not isinstance(payload, dict) or "state_dicts" not in payload or "cfg" not in payload:
        raise ConvertError(f"{path} is not a BPP workspace checkpoint (cfg, state_dicts, pickles)")
    if "model" not in payload["state_dicts"]:
        raise ConvertError(f"{path} holds no state_dicts.model")
    return payload


def checkpoint_tensors(state: dict[str, Any]) -> dict[str, Any]:
    """The tensors of a checkpoint's model state dict; the split info, and nothing else, dropped."""
    import torch

    others = sorted(k for k, v in state.items() if not isinstance(v, torch.Tensor))
    if set(others) - {SPLIT_INFO_KEY}:
        raise ConvertError(
            f"the model state holds non-tensors other than {SPLIT_INFO_KEY}: {others}"
        )
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


def check_tied(tensors: dict[str, Any], tied: dict[str, str]) -> None:
    """Every alias present, and equal to its canonical tensor, bit for bit."""
    import torch

    for alias, canonical in tied.items():
        if alias not in tensors or canonical not in tensors:
            raise ConvertError(f"tied tensor {alias} or its canonical {canonical} is missing")
        a, c = tensors[alias], tensors[canonical]
        if a.shape != c.shape or a.dtype != c.dtype or not torch.equal(a, c):
            raise ConvertError(
                f"{alias} differs from {canonical}, but the architecture uses one tensor for both"
            )


def dtype_name(tensor: Any) -> str:
    """The safetensors name of a torch tensor's dtype."""
    import torch

    names = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    return names.get(tensor.dtype, str(tensor.dtype))


def convert(
    ckpt: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    template: str | os.PathLike[str] | None = None,
    architecture: str = ARCHITECTURE,
    overwrite: bool = False,
) -> tuple[CheckReport, dict[str, Any]]:
    """Write `<out_dir>/model.safetensors` from the checkpoint at `ckpt`; `(check report, info)`.

    `info` lists what was dropped and where the checkpoint's configuration differs from the
    template's: the validator always runs the template's, so a difference is worth knowing.
    """
    from safetensors.torch import save_file

    out = Path(out_dir)
    target = out / WEIGHTS_FILENAME
    if target.exists() and not overwrite:
        raise ConvertError(f"{target} exists; pass --overwrite to replace it")
    config = load_config(template, architecture)
    expected = load_tensors(template, architecture)
    tied = config["tied_tensors"]

    payload = load_checkpoint(ckpt)
    state = payload["state_dicts"]["model"]
    tensors = checkpoint_tensors(state)
    check_tied(tensors, tied)
    # A copy of each: no two stored tensors may share memory, whatever the pickle held.
    kept = {k: v.detach().clone().contiguous() for k, v in tensors.items() if k not in tied}
    missing = sorted(set(expected) - set(kept))
    unexpected = sorted(set(kept) - set(expected))
    if missing or unexpected:
        raise ConvertError(
            f"the checkpoint is not this architecture: {len(missing)} tensors missing "
            f"({missing[:10]}), {len(unexpected)} unexpected ({unexpected[:10]})"
        )

    out.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        # One metadata entry: safetensors keeps metadata in a hash map, so two entries are
        # written in either order and the same checkpoint would convert to two different files.
        save_file(kept, str(partial), metadata=METADATA)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    info = {
        "checkpoint": str(ckpt),
        "dropped_non_tensors": sorted(set(state) - set(tensors)),
        "dropped_tied_aliases": len(tied),
        "config_differences": config_differences(payload["cfg"], config),
    }
    return check(target, template=template, architecture=architecture), info


def config_differences(cfg: Any, config: dict[str, Any]) -> list[str]:
    """Where the checkpoint's model, shape_meta and dataset configuration differ from the
    template's, as `section.path: checkpoint value != template value` lines."""
    from omegaconf import OmegaConf

    from .template import PROMPT_DATASET_OVERRIDES

    dataset = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    dataset.pop("shape_meta", None)
    dataset.update(PROMPT_DATASET_OVERRIDES)
    sections = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
        "dataset": dataset,
    }
    lines = []
    for section, value in sections.items():
        mine, theirs = _flatten(value), _flatten(config[section])
        for key in sorted(set(mine) | set(theirs)):
            if mine.get(key, "<absent>") != theirs.get(key, "<absent>"):
                lines.append(
                    f"{section}.{key}: {mine.get(key, '<absent>')!r} != "
                    f"{theirs.get(key, '<absent>')!r}"
                )
    return lines


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    if isinstance(node, list):
        out = {}
        for index, value in enumerate(node):
            out.update(_flatten(value, f"{prefix}{index}."))
        return out
    return {prefix[:-1]: node}
