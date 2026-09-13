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

import hashlib
import json
import re
import sys
from collections.abc import Collection
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

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


def installed_wheel_sha256(dist: Any) -> str | None:
    """The sha256 of the wheel a distribution was installed from, when that can be known.

    From its `direct_url.json` (PEP 610): the recorded archive hash (pip), a `#sha256=` fragment
    on the url, or the wheel file itself when it is still where it was installed from (uv records
    the path but no hash). An index install records none of these, and the pin cannot be confirmed.
    """
    try:
        text = dist.read_text("direct_url.json")
    except (OSError, ValueError):
        text = None
    if not text:
        return None
    try:
        doc = json.loads(text)
    except ValueError:
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
        if path.is_file() and path.suffix == ".whl":
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


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
