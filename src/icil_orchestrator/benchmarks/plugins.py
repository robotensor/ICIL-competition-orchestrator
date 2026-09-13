"""Finding the benchmarks installed beside the orchestrator, and refusing the wrong ones.

A benchmark advertises itself through the `icil.benchmarks` entry point group. Loading an entry
point imports code into the orchestrator's process, so what may be imported is part of the
contract: `spec.json` pins each benchmark by distribution, version and wheel sha256, and the pin is
checked from the installed distribution's metadata **before** anything is imported from it. A
benchmark the spec does not declare is listed but never imported.

Nothing here raises for a listing: an operator asking what is wrong is told everything at once.
`load` is the one call that refuses, and it says why.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from collections.abc import Collection
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..canon import sha256_file
from .api import ENTRY_POINT_GROUP, validate_plugin

#: Top-level modules of the simulator and model stacks a benchmark or a submission may bring.
#: None of them may be imported by loading a plugin: its pure half must run without them.
SIMULATOR_MODULES = frozenset({"sapien", "mujoco", "robosuite", "torch", "curobo", "isaacgym"})


class BenchmarkRefused(RuntimeError):
    """A benchmark is missing, does not match its pin, or is not a usable plugin."""


@dataclass
class Plugged:
    """One benchmark as found on this host: what the spec pins and what is installed."""

    name: str
    pin: dict[str, Any] | None
    entry_point: str | None = None
    distribution: str | None = None
    version: str | None = None
    wheel_sha256: str | None = None
    problems: list[str] = field(default_factory=list)
    #: Not problems: things an operator should know, such as a pin not filled in yet.
    notes: list[str] = field(default_factory=list)
    benchmark: Any = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "declared": self.pin is not None,
            "entry_point": self.entry_point,
            "distribution": self.distribution,
            "version": self.version,
            "wheel_sha256": self.wheel_sha256,
            "loaded": self.benchmark is not None,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def canonical_name(name: str) -> str:
    """A distribution name as the packaging tools compare them (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _entry_points() -> list[Any]:
    return list(metadata.entry_points(group=ENTRY_POINT_GROUP))


def _direct_url(dist: Any) -> dict[str, Any]:
    try:
        text = dist.read_text("direct_url.json")
    except (OSError, ValueError):
        text = None
    try:
        doc = json.loads(text) if text else {}
    except ValueError:
        doc = {}
    return doc if isinstance(doc, dict) else {}


def installed_wheel_sha256(dist: Any) -> str | None:
    """The sha256 of the wheel a distribution was installed from, when that can be known.

    From its `direct_url.json` (PEP 610): the recorded archive hash (pip), a `#sha256=` fragment
    on the url, or the wheel file itself when it is still where it was installed from (uv records
    the path but no hash) - and then only if that wheel's RECORD agrees with what was installed,
    since a wheel rebuilt in place without reinstalling says nothing about the installed code. An
    index install records none of these, and the pin cannot be confirmed.
    """
    doc = _direct_url(dist)
    if not doc:
        return None
    archive = doc.get("archive_info") or {}
    hashes = archive.get("hashes") or {}
    if isinstance(hashes.get("sha256"), str):
        return hashes["sha256"].lower()
    legacy = archive.get("hash")
    if isinstance(legacy, str) and legacy.startswith("sha256="):
        return legacy.split("=", 1)[1].lower()
    url = str(doc.get("url") or "")
    parsed = urlparse(url)
    fragment = dict(p.split("=", 1) for p in parsed.fragment.split("&") if "=" in p)
    if "sha256" in fragment:
        return fragment["sha256"].lower()
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        if path.is_file() and path.suffix == ".whl" and _wheel_is_installed(path, dist):
            return sha256_file(path)
    return None


def _record_hashes(lines: str) -> dict[str, str]:
    """`{path: "sha256=<urlsafe b64>"}` for the hashed entries of a RECORD file."""
    out: dict[str, str] = {}
    for line in lines.splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) == 3 and parts[1].startswith("sha256="):
            out[parts[0]] = parts[1]
    return out


def _wheel_is_installed(wheel: Path, dist: Any) -> bool:
    """Every file the wheel's RECORD hashes (scripts and data aside, which an installer moves) is
    installed with that hash."""
    installed = {str(f): f"{f.hash.mode}={f.hash.value}" for f in dist.files or () if f.hash}
    try:
        with zipfile.ZipFile(wheel) as zf:
            names = [n for n in zf.namelist() if n.endswith(".dist-info/RECORD")]
            if len(names) != 1:
                return False
            record = _record_hashes(zf.read(names[0]).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError):
        return False
    wanted = {k: v for k, v in record.items() if ".data/" not in k}
    return bool(wanted) and all(installed.get(k) == v for k, v in wanted.items())


def module_origin_problem(ep: Any, dist: Any) -> str | None:
    """Why the module `ep` names would not be imported from the distribution that was pinned.

    Entry points import by module name, so a same-named module earlier on `sys.path` - the cwd
    under `python -m`, a benchmark checkout - would run in place of the pinned wheel while the
    metadata checks passed. Resolved without importing anything.
    """
    module = str(getattr(ep, "module", "") or ep.value.split(":")[0]).strip()
    top = module.split(".")[0]
    try:
        found = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return None  # loading reports the import failure with its own message
    if found is None:
        return None
    if found.origin and found.has_location:
        paths = [Path(found.origin).resolve()]
    else:
        paths = [Path(p).resolve() for p in found.submodule_search_locations or ()]
    if not paths:
        return f"{top} is not importable from a file, so it cannot be tied to {dist.name}"

    direct = _direct_url(dist)
    if (direct.get("dir_info") or {}).get("editable"):
        root = Path(unquote(urlparse(str(direct.get("url") or "")).path)).resolve()
        belongs = all(path.is_relative_to(root) for path in paths)
    elif dist.files is not None:
        files = {Path(dist.locate_file(f)).resolve() for f in dist.files}
        belongs = all(
            path in files or (path.is_dir() and any(f.is_relative_to(path) for f in files))
            for path in paths
        )
    else:
        site = Path(dist.locate_file("")).resolve()
        belongs = all(path.is_relative_to(site) for path in paths)
    if belongs:
        return None
    where = ", ".join(str(p) for p in paths)
    return f"{top} resolves to {where}, which is not a file of {dist.name}; refusing to import it"


def record_problems(dist: Any) -> list[str]:
    """Installed files whose bytes no longer match the hash their RECORD gives them."""
    problems = []
    for f in dist.files or ():
        if not f.hash or f.hash.mode != "sha256":
            continue
        path = Path(dist.locate_file(f))
        try:
            digest = hashlib.sha256(path.read_bytes()).digest()
        except OSError:
            problems.append(f"{f} is in its RECORD but missing")
            continue
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != f.hash.value:
            problems.append(
                f"{f} differs from its RECORD hash: the installed files are not the pinned wheel"
            )
    return problems


def pin_problems(plugged: Plugged) -> None:
    """Compare what is installed with what the spec pins; record problems and notes in place."""
    pin = plugged.pin
    if pin is None:
        plugged.problems.append(
            "not declared in spec.json `benchmarks`; an undeclared benchmark is never imported"
        )
        return
    wanted = str(pin.get("distribution") or "")
    if plugged.distribution is None or canonical_name(plugged.distribution) != canonical_name(
        wanted
    ):
        plugged.problems.append(
            f"provided by distribution {plugged.distribution!r}, but spec.json pins {wanted!r}"
        )
        return
    version = pin.get("version")
    if version is None:
        plugged.notes.append("version not pinned in spec.json yet")
    elif plugged.version != version:
        plugged.problems.append(f"version {plugged.version} is not the pinned {version}")
    wheel = pin.get("wheel_sha256")
    if wheel is None:
        plugged.notes.append("wheel sha256 not pinned in spec.json yet")
    elif plugged.wheel_sha256 is None:
        plugged.problems.append(
            "the pinned wheel sha256 cannot be confirmed: the distribution records no wheel hash "
            "(install it from the pinned wheel file)"
        )
    elif plugged.wheel_sha256 != str(wheel).lower():
        plugged.problems.append(
            f"installed from wheel sha256 {plugged.wheel_sha256[:16]}..., "
            f"not the pinned {str(wheel)[:16]}..."
        )


def discover(spec: Any, *, load: bool | Collection[str] = False) -> dict[str, Plugged]:
    """Every benchmark the spec declares or this host advertises, keyed by name.

    `load` (True, or the names to load) imports each one whose pin matches and checks it
    structurally; a benchmark with a pin problem is never imported.
    """
    pins = spec.benchmarks
    found: dict[str, Plugged] = {}
    by_name: dict[str, list[Any]] = {}
    for ep in _entry_points():
        by_name.setdefault(ep.name, []).append(ep)

    for name, eps in sorted(by_name.items()):
        ep = eps[0]
        dist = getattr(ep, "dist", None)
        plugged = Plugged(
            name=name,
            pin=pins.get(name),
            entry_point=ep.value,
            distribution=getattr(dist, "name", None) if dist is not None else None,
            version=getattr(dist, "version", None) if dist is not None else None,
            wheel_sha256=installed_wheel_sha256(dist) if dist is not None else None,
        )
        if len(eps) > 1:
            names = sorted({str(getattr(getattr(e, "dist", None), "name", "?")) for e in eps})
            plugged.problems.append(
                f"advertised by {len(eps)} entry points ({', '.join(names)}); remove all but one"
            )
        pin_problems(plugged)
        wanted = load is True or (not isinstance(load, bool) and name in load)
        if wanted and plugged.ok and dist is not None:
            origin = module_origin_problem(ep, dist)
            if origin:
                plugged.problems.append(origin)
            elif (plugged.pin or {}).get("wheel_sha256") is not None:
                # A wheel pin holds the bytes that are imported, not only the metadata beside them.
                plugged.problems.extend(record_problems(dist))
        if wanted and plugged.ok:
            _load_into(plugged, ep)
        found[name] = plugged

    for name, pin in pins.items():
        if name not in found:
            found[name] = Plugged(
                name=name,
                pin=pin,
                problems=[
                    f"not installed: install {pin.get('distribution') or 'its distribution'}, "
                    f"which advertises {name!r} in the {ENTRY_POINT_GROUP!r} entry point group"
                ],
            )
    return dict(sorted(found.items()))


def _load_into(plugged: Plugged, ep: Any) -> None:
    before = set(sys.modules)
    try:
        loaded = ep.load()
    except Exception as exc:  # noqa: BLE001 - a plugin's import failure is reported, not raised
        plugged.problems.append(f"{ep.value} did not import: {type(exc).__name__}: {exc}")
        return
    leaked = sorted({m.split(".")[0] for m in set(sys.modules) - before} & SIMULATOR_MODULES)
    if leaked:
        plugged.problems.append(
            f"importing {ep.value} imported {', '.join(leaked)}; a plugin's pure half must not"
        )
    candidate = loaded if hasattr(loaded, "api_version") else getattr(loaded, "BENCHMARK", None)
    if candidate is None:
        plugged.problems.append(
            f"{ep.value} exposes no benchmark: expected the plugin object, or a module with a "
            "`BENCHMARK` attribute"
        )
        return
    errors = validate_plugin(candidate)
    plugged.problems.extend(errors)
    if not errors and candidate.id != plugged.name:
        plugged.problems.append(
            f"{ep.value} is advertised as {plugged.name!r} but calls itself {candidate.id!r}"
        )
    if plugged.ok:
        plugged.benchmark = candidate


def load(spec: Any, name: str) -> Any:
    """The plugin object for `name`, or `BenchmarkRefused` naming every reason it cannot be used."""
    plugged = discover(spec, load=(name,)).get(name)
    if plugged is None:
        raise BenchmarkRefused(f"{name}: not declared in spec.json and not installed")
    if not plugged.ok:
        raise BenchmarkRefused(f"{name}: " + "; ".join(plugged.problems))
    return plugged.benchmark
