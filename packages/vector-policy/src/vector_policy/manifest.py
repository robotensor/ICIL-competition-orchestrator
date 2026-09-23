"""`policy.yaml`: what a competitor's repository serves, and how.

    api: 1                              # required; the version of this schema and of the protocol
    policy: my_policy.policy:MyPolicy   # required; module:Class, importable from the repo root
    kwargs: {checkpoint: weights.pt}    # optional; passed to the constructor
    requirements: requirements.txt      # optional; relative to the repo root, inside it
    benchmarks: [robotwin]              # optional; the benchmarks the policy is meant for

The file lives at the root of the repository. It is read with PyYAML's safe loader, so a tag that
would construct a Python object is refused rather than run. A key given twice and an unknown key
are refused too: a typo in a submission should fail when it is checked, not be ignored until a
duel. Every problem is reported at once, in one `ManifestError`, which is the only exception
`load` raises: the file is untrusted, and whoever checks it should need to catch nothing else.
"""

from __future__ import annotations

import os
import re
import reprlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ManifestError

#: The one `api` this version of the distribution serves.
API_VERSION = 1

#: The file name a competitor's repository holds its manifest under.
FILENAME = "policy.yaml"

#: The largest manifest read. One names a class and a few constructor arguments.
MAX_BYTES = 1 << 20

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_POLICY = re.compile(rf"{_IDENTIFIER}(\.{_IDENTIFIER})*:{_IDENTIFIER}")
_KEYS = ("api", "policy", "kwargs", "requirements", "benchmarks")

#: What a problem quotes of a value: an excerpt, since YAML aliases let a small file hold a value
#: whose full repr would not fit in memory.
_EXCERPT = reprlib.Repr()
_EXCERPT.maxlevel = 2
_EXCERPT.maxstring = _EXCERPT.maxother = _EXCERPT.maxlong = 60
_EXCERPT.maxlist = _EXCERPT.maxtuple = _EXCERPT.maxdict = _EXCERPT.maxset = 6
#: The longest message of a parser error quoted in a problem.
_PARSER_CHARS = 1000


@dataclass(frozen=True)
class Manifest:
    """A valid `policy.yaml`. `root` is the directory holding it: the competitor's repository."""

    path: Path
    api: int
    policy: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    requirements: str | None = None
    benchmarks: tuple[str, ...] = ()

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def module(self) -> str:
        return self.policy.partition(":")[0]

    @property
    def attribute(self) -> str:
        return self.policy.partition(":")[2]

    @property
    def requirements_path(self) -> Path | None:
        return None if self.requirements is None else self.root / self.requirements


def load(path: str | os.PathLike[str]) -> Manifest:
    """The manifest at `path`, validated; `ManifestError` listing every problem otherwise."""
    path = Path(path).absolute()
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        text = raw.decode("utf-8")
    except (OSError, ValueError) as exc:  # ValueError: not UTF-8, or a NUL in the path
        raise ManifestError(str(path), [f"cannot be read: {exc}"]) from None
    if len(raw) > MAX_BYTES:
        raise ManifestError(str(path), [f"is larger than {MAX_BYTES} bytes"])
    try:
        data = yaml.load(text, Loader=_SafeUniqueLoader)
    except RecursionError:
        raise ManifestError(str(path), ["is not valid YAML: nested too deeply"]) from None
    except Exception as exc:  # YAMLError, and a value the safe loader cannot build: 2026-02-30
        message = str(exc)
        if len(message) > _PARSER_CHARS:
            message = message[:_PARSER_CHARS] + " [...]"
        raise ManifestError(str(path), [f"is not valid YAML: {message}"]) from None
    return validate(data, path)


def validate(data: Any, path: str | os.PathLike[str]) -> Manifest:
    """`data`, as parsed from the manifest at `path`, checked against the schema."""
    path = Path(path).absolute()
    if not isinstance(data, dict):
        raise ManifestError(str(path), [f"must be a mapping, not {type(data).__name__}"])
    problems: list[str] = []

    unknown = [key for key in data if key not in _KEYS]
    if unknown:
        problems.append(
            f"unknown key(s) {', '.join(map(_excerpt, unknown))}; allowed: {', '.join(_KEYS)}"
        )

    api = data.get("api")
    if "api" not in data:
        problems.append("api: required")
    elif type(api) is not int or api != API_VERSION:
        problems.append(f"api: must be {API_VERSION}, not {_excerpt(api)}")

    policy = data.get("policy")
    if "policy" not in data:
        problems.append("policy: required, as module:Class")
    elif not isinstance(policy, str) or not _POLICY.fullmatch(policy):
        problems.append(
            f"policy: must be module:Class, such as pkg.module:MyPolicy, not {_excerpt(policy)}"
        )

    kwargs = data.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        problems.append(f"kwargs: must be a mapping, not {type(kwargs).__name__}")
        kwargs = {}
    else:
        bad = [key for key in kwargs if not isinstance(key, str) or not key.isidentifier()]
        if bad:
            problems.append(
                f"kwargs: keys must be identifiers, not {', '.join(map(_excerpt, bad))}"
            )

    requirements = data.get("requirements")
    if requirements is not None:
        problems.extend(_requirements_problems(requirements, path.parent))

    benchmarks = data.get("benchmarks", [])
    if benchmarks is None:
        benchmarks = []
    if not isinstance(benchmarks, list) or not all(isinstance(b, str) and b for b in benchmarks):
        problems.append(f"benchmarks: must be a list of names, not {_excerpt(benchmarks)}")
        benchmarks = []

    if problems:
        raise ManifestError(str(path), problems)
    return Manifest(
        path=path,
        api=api,
        policy=policy,
        kwargs=dict(kwargs),
        requirements=requirements,
        benchmarks=tuple(benchmarks),
    )


_MERGE_TAG = "tag:yaml.org,2002:merge"


class _SafeUniqueLoader(yaml.SafeLoader):
    """`yaml.safe_load`, refusing a key given twice instead of keeping the last one silently.

    Only the keys written in a mapping count: a merge key (`<<: *anchor`) is left to the base
    class, and a key written beside it overrides the merged one, as YAML says it does.
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # unhashable: the base class refuses it with its own message
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None, None, f"key {key!r} is given twice", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _requirements_problems(requirements: Any, root: Path) -> list[str]:
    if not isinstance(requirements, str) or not requirements:
        return [f"requirements: must be a path, not {_excerpt(requirements)}"]
    shown = _excerpt(requirements)
    if "\0" in requirements:
        return [f"requirements: {shown} holds a NUL character"]
    if os.path.isabs(requirements) or requirements.startswith("~"):
        return [f"requirements: {shown} must be relative to the repository root"]
    try:
        real_root = root.resolve()
        target = (root / requirements).resolve()
        if target != real_root and real_root not in target.parents:
            return [f"requirements: {shown} leaves the repository"]
        if not target.is_file():
            return [f"requirements: {shown} is not a file in the repository"]
    except (OSError, RuntimeError) as exc:  # RuntimeError: a symlink loop, before Python 3.13
        return [f"requirements: {shown} cannot be resolved: {_excerpt(str(exc))}"]
    return []


def _excerpt(value: Any) -> str:
    return _EXCERPT.repr(value)
