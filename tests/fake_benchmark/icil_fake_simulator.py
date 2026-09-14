"""A stand-in for a simulator stack (SAPIEN, MuJoCo): only a benchmark's command half imports it.

Tests assert this module never appears in the orchestrator's process.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

NAME = "icil-fake-simulator"


def digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
