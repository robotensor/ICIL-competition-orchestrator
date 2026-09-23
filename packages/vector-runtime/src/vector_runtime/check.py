"""Is this `model.safetensors` exactly the pinned architecture's tensors? Decided from the header.

    vector-runtime check --weights DIR_OR_FILE [--template DIR] [--max-bytes N] [--no-hash]

The standard library only: no torch, no safetensors, no numpy. Nothing in the file is executed or
deserialized beyond its JSON header, which `vector_runtime.header` parses by hand and strictly. The
file passes when

- it is a regular file of at most `max_file_bytes` (8 GiB by default), with a sane header;
- its header is well formed and its tensors tile the data section exactly;
- it holds exactly the template's tensor keys, each with the template's shape and dtype.

Normalizer statistics are loaded dynamically by the network, so a strict `load_state_dict` would
not notice one missing or added: that is why the key set is checked here, exactly, and why the
loader (`vector_runtime.model`) also refuses non-finite or zero-scale statistics, which a header
cannot show.

The report carries `weights_sha256`, the sha256 of the file's bytes: the identity a result is
recorded under.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import ARCHITECTURE, WEIGHTS_FILENAME
from .header import DEFAULT_MAX_HEADER_BYTES, HeaderError, read_header
from .template import TemplateError, load_tensors

#: The largest file checked unless the caller says otherwise. The real one is ~2.2 GB.
DEFAULT_MAX_FILE_BYTES = 8 * 1024**3
#: How many missing, unexpected or mismatched tensors an error lists by name.
LISTED = 20
_CHUNK = 8 * 1024 * 1024


@dataclass
class CheckReport:
    path: str
    architecture: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    file_bytes: int | None = None
    header_bytes: int | None = None
    tensor_count: int | None = None
    param_count: int | None = None
    metadata: dict[str, str] | None = None
    weights_sha256: str | None = None
    other_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, **asdict(self)}


def weights_file(path: str | os.PathLike[str]) -> Path:
    """`path` itself, or `path/model.safetensors` when it is a directory."""
    path = Path(path)
    return path / WEIGHTS_FILENAME if path.is_dir() else path


def check(
    weights: str | os.PathLike[str],
    *,
    template: str | os.PathLike[str] | None = None,
    architecture: str = ARCHITECTURE,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    compute_sha256: bool = True,
) -> CheckReport:
    """Check the weights at `weights` (a file, or a directory holding `model.safetensors`)."""
    path = weights_file(weights)
    report = CheckReport(path=str(path), architecture=architecture)
    if Path(weights).is_dir():
        report.other_files = sorted(
            entry.name for entry in Path(weights).iterdir() if entry.name != WEIGHTS_FILENAME
        )
        if report.other_files:
            report.warnings.append(
                f"the directory holds more than {WEIGHTS_FILENAME}: {', '.join(report.other_files)}"
            )
    try:
        expected = load_tensors(template, architecture)
    except TemplateError as exc:
        report.errors.append(str(exc))
        return report

    try:
        info = path.stat()  # follows a symbolic link, as a Hugging Face snapshot holds them
    except OSError as exc:
        report.errors.append(f"cannot read {path}: {exc.strerror or exc}")
        return report
    if not stat.S_ISREG(info.st_mode):
        report.errors.append(f"{path} is not a regular file")
        return report
    report.file_bytes = info.st_size
    if info.st_size > max_file_bytes:
        report.errors.append(f"the file is {info.st_size} bytes; at most {max_file_bytes}")
        return report

    try:
        with path.open("rb") as handle:
            header = read_header(handle, info.st_size, max_header_bytes=max_header_bytes)
    except HeaderError as exc:
        report.errors.append(f"not a valid safetensors file: {exc}")
    except OSError as exc:
        report.errors.append(f"cannot read {path}: {exc.strerror or exc}")
        return report
    else:
        report.header_bytes = header.header_bytes
        report.tensor_count = len(header.tensors)
        report.param_count = header.param_count
        report.metadata = header.metadata
        report.errors.extend(compare(header.tensors, expected))

    if compute_sha256:
        try:
            report.weights_sha256 = sha256_file(path)
        except OSError as exc:
            report.errors.append(f"cannot hash {path}: {exc.strerror or exc}")
    return report


#: The name `vector_orchestrator.duel.weights_runtime` calls it by.
check_weights = check


def compare(tensors: dict[str, Any], expected: dict[str, dict[str, Any]]) -> list[str]:
    """Every way the header's tensors differ from the template's, as error messages."""
    errors: list[str] = []
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    for key in missing[:LISTED]:
        errors.append(f"tensor {key} is missing")
    for key in unexpected[:LISTED]:
        errors.append(f"tensor {key} is not part of the architecture")
    mismatched = 0
    for key in sorted(set(tensors) & set(expected)):
        got, want = tensors[key], expected[key]
        problems = []
        if list(got.shape) != list(want["shape"]):
            problems.append(f"shape {list(got.shape)} != {list(want['shape'])}")
        if got.dtype != want["dtype"]:
            problems.append(f"dtype {got.dtype} != {want['dtype']}")
        if problems:
            mismatched += 1
            if mismatched <= LISTED:
                errors.append(f"tensor {key}: {', '.join(problems)}")
    if len(missing) > LISTED or len(unexpected) > LISTED or mismatched > LISTED:
        errors.append(
            f"in all: {len(missing)} missing, {len(unexpected)} unexpected and {mismatched} "
            f"mismatched of the {len(expected)} tensors the architecture has"
        )
    return errors


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
