"""A safetensors file's header, parsed by hand: the standard library only, no tensor data read.

The format: an 8-byte little-endian unsigned length N, then N bytes of UTF-8 JSON, then the data.
The JSON maps each tensor name to `{"dtype", "shape", "data_offsets": [begin, end]}`, offsets
relative to the start of the data, plus an optional `"__metadata__"` of string to string.

The file is untrusted, so the parse is strict where a lenient one would let two readers disagree
about what the file holds: a duplicated key, a NaN, an unknown dtype, a byte range that does not
match its shape, a gap or an overlap between tensors, or data past the last tensor are all
refused. What is accepted is exactly what `safetensors` itself would load, tensor for tensor.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO

#: Bytes per element of every dtype safetensors defines.
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
METADATA_KEY = "__metadata__"
LENGTH_BYTES = 8
#: The largest header read unless the caller says otherwise. The real one is ~0.1 MB.
DEFAULT_MAX_HEADER_BYTES = 16 * 1024 * 1024
#: The most dimensions a tensor may declare.
MAX_RANK = 16


class HeaderError(ValueError):
    """The bytes are not a well-formed safetensors header."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n


@dataclass(frozen=True)
class Header:
    """A parsed header: its JSON length, the tensors, the metadata and the data section's size."""

    header_bytes: int
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str] | None = None
    data_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return LENGTH_BYTES + self.header_bytes + self.data_bytes

    @property
    def param_count(self) -> int:
        return sum(t.numel for t in self.tensors.values())


def read_header(
    handle: BinaryIO, file_size: int, *, max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES
) -> Header:
    """The header of the safetensors file open as `handle` (at offset 0), `file_size` bytes long.

    Raises `HeaderError` for anything that is not a well-formed file of exactly that size.
    """
    prefix = handle.read(LENGTH_BYTES)
    if len(prefix) != LENGTH_BYTES:
        raise HeaderError(f"the file is {file_size} bytes, too short for a safetensors header")
    (length,) = struct.unpack("<Q", prefix)
    if length < 2:
        raise HeaderError(f"the header length is {length} bytes, too short for a JSON object")
    if length > max_header_bytes:
        raise HeaderError(f"the header is {length} bytes; at most {max_header_bytes} are read")
    if length > file_size - LENGTH_BYTES:
        raise HeaderError(
            f"the header claims {length} bytes but the file has {file_size - LENGTH_BYTES} after "
            "the length"
        )
    raw = handle.read(length)
    if len(raw) != length:
        raise HeaderError(f"the header was cut short: {len(raw)} of {length} bytes")
    return parse_header(raw, file_size)


def parse_header(raw: bytes, file_size: int) -> Header:
    """The header whose JSON bytes are `raw`, in a file of `file_size` bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HeaderError(f"the header is not UTF-8: {exc}") from None
    if not text.startswith("{"):
        raise HeaderError("the header does not start with a JSON object")
    try:
        document = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_refuse_constant
        )
    except HeaderError:
        raise
    except (ValueError, RecursionError) as exc:
        raise HeaderError(f"the header is not valid JSON: {exc}") from None
    if not isinstance(document, dict):
        raise HeaderError("the header is not a JSON object")

    metadata = document.pop(METADATA_KEY, None)
    if metadata is not None:
        if not isinstance(metadata, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
        ):
            raise HeaderError(f"{METADATA_KEY} must map strings to strings")

    tensors = {name: _tensor(name, entry) for name, entry in document.items()}
    data_bytes = file_size - LENGTH_BYTES - len(raw)
    _check_layout(tensors, data_bytes)
    return Header(header_bytes=len(raw), tensors=tensors, metadata=metadata, data_bytes=data_bytes)


def _tensor(name: str, entry: Any) -> TensorInfo:
    if not name:
        raise HeaderError("a tensor has an empty name")
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise HeaderError(f"{name}: expected exactly dtype, shape and data_offsets")
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    if dtype not in DTYPE_SIZES:
        raise HeaderError(f"{name}: unknown dtype {dtype!r}")
    if not isinstance(shape, list) or len(shape) > MAX_RANK or not all(_count(d) for d in shape):
        raise HeaderError(f"{name}: shape must be a list of at most {MAX_RANK} non-negative ints")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(_count(o) for o in offsets):
        raise HeaderError(f"{name}: data_offsets must be two non-negative ints")
    begin, end = offsets
    info = TensorInfo(dtype=dtype, shape=tuple(shape), begin=begin, end=end)
    if end < begin or end - begin != info.numel * DTYPE_SIZES[dtype]:
        raise HeaderError(
            f"{name}: data_offsets {offsets} hold {end - begin} bytes; a {dtype} tensor of shape "
            f"{shape} is {info.numel * DTYPE_SIZES[dtype]}"
        )
    return info


def _check_layout(tensors: dict[str, TensorInfo], data_bytes: int) -> None:
    """The tensors tile the data section exactly: no gap, no overlap, nothing after the last."""
    cursor = 0
    for name, info in sorted(tensors.items(), key=lambda item: (item[1].begin, item[1].end)):
        if info.begin != cursor:
            what = "overlaps the tensor before it" if info.begin < cursor else "leaves a gap"
            raise HeaderError(f"{name}: starts at byte {info.begin} and {what} (expected {cursor})")
        cursor = info.end
    if cursor != data_bytes:
        raise HeaderError(
            f"the tensors hold {cursor} bytes of data but the file has {data_bytes} after the header"
        )


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise HeaderError(f"the header has the key {key!r} twice")
        out[key] = value
    return out


def _refuse_constant(name: str) -> Any:
    raise HeaderError(f"the header holds {name}, which is not JSON")
