"""Synthetic safetensors files, written by hand: the host tests need neither torch nor safetensors."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from bpp_runtime import ARCHITECTURE
from bpp_runtime.header import DTYPE_SIZES

#: A small architecture: a weight, a bias, a zero-size dummy and a normalizer statistic.
TENSORS = {
    "_dummy_variable": ("F32", [0]),
    "model.weight": ("F32", [4, 3]),
    "model.bias": ("F32", [4]),
    "normalizer.params_dict.action.scale": ("F32", [20]),
}


def numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def safetensors_bytes(
    tensors: dict[str, tuple[str, list[int]]] | None = None,
    *,
    metadata: dict[str, str] | None = None,
    header: bytes | None = None,
    extra: bytes = b"",
    length: int | None = None,
) -> bytes:
    """A safetensors file holding zeros for `tensors`; `header`, `extra` bytes after the data
    and a false `length` field make it malformed on purpose."""
    tensors = TENSORS if tensors is None else tensors
    document: dict[str, Any] = {}
    if metadata is not None:
        document["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = numel(shape) * DTYPE_SIZES[dtype]
        document[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(document).encode() if header is None else header
    raw += b" " * (-len(raw) % 8)
    prefix = struct.pack("<Q", len(raw) if length is None else length)
    return prefix + raw + bytes(offset) + extra


def write_template(
    directory: Path, tensors: dict[str, tuple[str, list[int]]] | None = None
) -> Path:
    """A template directory whose tensor manifest is `tensors`."""
    tensors = TENSORS if tensors is None else tensors
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {name: {"shape": shape, "dtype": dtype} for name, (dtype, shape) in tensors.items()}
    (directory / f"{ARCHITECTURE}.tensors.json").write_text(json.dumps(manifest))
    return directory
