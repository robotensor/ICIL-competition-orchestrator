"""The pinned architecture a submission is held to: `arch/<name>.cfg.json` and `.tensors.json`.

Both files are generated once, from the base checkpoint, by `vector-runtime template`, and are
committed: a submission never brings configuration of its own.

`<name>.tensors.json` is `{key: {"shape": [...], "dtype": "F32"}}` for every tensor
`model.safetensors` must hold - no more, no fewer. `<name>.cfg.json` holds:

- `model`: the checkpoint's resolved model configuration, every `_target_` included; the network
  is `hydra.utils.instantiate(model)`.
- `shape_meta`: the task's observation and action layout, which the prompt dataset is handed.
- `dataset`: the task's resolved dataset configuration, with the three settings the reference
  adapter overrides to build a prompt from one demonstration (`dataset_path`, `cache_dir`,
  `val_ratio`); `shape_meta` is passed beside it, as the reference does.
- `obs_horizons`, `n_obs_steps`, `action_horizon`, `exec_action_horizon`, `action_dim`.
- `tied_tensors`: `{alias: canonical}`. The network uses one module in two places - the prompt
  and the receding observation encoders are the same module, and the right wrist camera shares
  the left wrist camera's vision model - so its state dict names one tensor under several keys.
  `model.safetensors` stores the canonical key only; the loader restores every alias from it.
- `source`: which checkpoint and which behavior_prompting source the template came from.

This module's loaders use the standard library only. `generate` needs the model extra.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import ARCHITECTURE

#: The template directory shipped with this package.
ARCH_DIR = Path(__file__).resolve().parent / "arch"
#: An environment variable naming another template directory, for the check and the policy alike.
TEMPLATE_ENV = "VECTOR_RUNTIME_TEMPLATE"
TEMPLATE_FORMAT = 1
#: What the reference adapter executes of each predicted chunk (its deploy.yml default).
DEFAULT_EXEC_ACTION_HORIZON = 12
#: The reference adapter builds a prompt with these dataset settings, whatever training used.
PROMPT_DATASET_OVERRIDES = {"dataset_path": None, "cache_dir": None, "val_ratio": 0.0}
#: Every tensor whose key starts with this is a normalizer statistic, loaded dynamically.
NORMALIZER_PREFIX = "normalizer."
#: Where the vendored behavior_prompting source came from.
BEHAVIOR_PROMPTING_SOURCE = {
    "repository": "louis392/BRL1",
    "revision": "90de4af4d77e559f67c1142d0c4dcbcd3671dcdc",
    "file": "code/behavior_prompting_snapshot.tar.gz",
    "sha256": "07a8acccb08e40a4fe5337cf0af5e56e3af19ed0189d0eba63e21771ebbf5e4c",
}


class TemplateError(ValueError):
    """A template file is missing or malformed."""


def template_dir(path: str | os.PathLike[str] | None = None) -> Path:
    """`path`; else `$VECTOR_RUNTIME_TEMPLATE`; else the template shipped with this package."""
    if path is not None:
        return Path(path)
    return Path(os.environ.get(TEMPLATE_ENV) or ARCH_DIR)


def config_path(path: str | os.PathLike[str] | None = None, name: str = ARCHITECTURE) -> Path:
    return template_dir(path) / f"{name}.cfg.json"


def tensors_path(path: str | os.PathLike[str] | None = None, name: str = ARCHITECTURE) -> Path:
    return template_dir(path) / f"{name}.tensors.json"


def load_config(path: str | os.PathLike[str] | None = None, name: str = ARCHITECTURE) -> dict:
    """The architecture's configuration, as `cfg.json` holds it."""
    config = _read_json(config_path(path, name))
    if config.get("architecture") != name or config.get("template_format") != TEMPLATE_FORMAT:
        raise TemplateError(
            f"{config_path(path, name)} is not a format-{TEMPLATE_FORMAT} template of {name}"
        )
    return config


def load_tensors(
    path: str | os.PathLike[str] | None = None, name: str = ARCHITECTURE
) -> dict[str, dict[str, Any]]:
    """`{key: {"shape": [...], "dtype": ...}}`: every tensor `model.safetensors` must hold."""
    tensors = _read_json(tensors_path(path, name))
    for key, entry in tensors.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"shape", "dtype"}
            or not isinstance(entry["shape"], list)
            or not isinstance(entry["dtype"], str)
        ):
            raise TemplateError(f"{tensors_path(path, name)}: malformed entry for {key}")
    return tensors


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TemplateError(f"cannot read the template {path}: {exc}") from None
    if not isinstance(data, dict):
        raise TemplateError(f"{path} does not hold a JSON object")
    return data


def write_json(path: Path, data: Any) -> str:
    """Write `data` as sorted, indented JSON with a final newline; its sha256."""
    import hashlib

    text = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- generation (torch) ---------------------------------------------------------------------


def generate(
    ckpt: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    name: str = ARCHITECTURE,
    exec_action_horizon: int = DEFAULT_EXEC_ACTION_HORIZON,
) -> dict[str, Any]:
    """Write `<out_dir>/<name>.cfg.json` and `.tensors.json` from the checkpoint at `ckpt`.

    The configuration is the checkpoint's own, resolved. The tensor manifest and the tied keys
    come from the network that configuration builds, cross-checked against the checkpoint: every
    key the network has and nothing else (normalizer statistics aside, which the network only
    gets when they are loaded), every shape equal, and every alias equal to its canonical tensor.
    """
    from omegaconf import OmegaConf

    from . import convert, model

    ckpt = Path(ckpt)
    out = Path(out_dir)
    payload = convert.load_checkpoint(ckpt)
    cfg = payload["cfg"]
    state = payload["state_dicts"]["model"]

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    dataset_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    dataset_cfg.pop("shape_meta", None)  # handed over separately, as the reference adapter does
    dataset_cfg.update(PROMPT_DATASET_OVERRIDES)

    obs_horizons = {key: int(value["horizon"]) for key, value in shape_meta["obs"].items()}
    action_horizon = int(shape_meta["action"]["horizon"])
    if not 1 <= exec_action_horizon <= action_horizon:
        raise TemplateError(f"exec_action_horizon must be in [1, {action_horizon}]")

    network = model.build_model(model_cfg)
    live = model.state_keys(network)
    tied = model.tied_tensors(network)
    tensors = convert.checkpoint_tensors(state)
    normalizer = sorted(k for k in tensors if k.startswith(NORMALIZER_PREFIX))
    if not normalizer:
        raise TemplateError("the checkpoint holds no normalizer statistics")
    network_keys = set(tensors) - set(normalizer)
    if network_keys != set(live):
        raise TemplateError(
            f"the checkpoint and the network it configures disagree: missing "
            f"{sorted(set(live) - network_keys)[:10]}, unexpected {sorted(network_keys - set(live))[:10]}"
        )
    for key, shape in live.items():
        if tuple(tensors[key].shape) != shape:
            raise TemplateError(f"{key}: checkpoint shape {tuple(tensors[key].shape)} != {shape}")
    convert.check_tied(tensors, tied)

    stored = [k for k in tensors if k not in tied]
    manifest = {
        key: {"shape": list(tensors[key].shape), "dtype": convert.dtype_name(tensors[key])}
        for key in sorted(stored)
    }
    config = {
        "architecture": name,
        "template_format": TEMPLATE_FORMAT,
        "source": {
            "checkpoint": ckpt.name,
            "checkpoint_sha256": convert.sha256_file(ckpt),
            "behavior_prompting": BEHAVIOR_PROMPTING_SOURCE,
        },
        "model": model_cfg,
        "shape_meta": shape_meta,
        "dataset": dataset_cfg,
        "obs_horizons": obs_horizons,
        "n_obs_steps": max(obs_horizons.values()),
        "action_horizon": action_horizon,
        "action_dim": int(shape_meta["action"]["shape"][0]),
        "exec_action_horizon": int(exec_action_horizon),
        "tied_tensors": dict(sorted(tied.items())),
    }
    out.mkdir(parents=True, exist_ok=True)
    return {
        "architecture": name,
        "tensor_count": len(manifest),
        "param_count": sum(_numel(entry["shape"]) for entry in manifest.values()),
        "tied_aliases": len(tied),
        "normalizer_tensors": len(normalizer),
        "files": {
            str(config_path(out, name)): write_json(config_path(out, name), config),
            str(tensors_path(out, name)): write_json(tensors_path(out, name), manifest),
        },
    }


def _numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n
