"""Benchmarks that live outside this repository.

`api.py` says what a benchmark distribution must expose. Nothing in this package imports a
benchmark's simulator: a plugin's pure half is called in-process, and everything else runs as the
argv its command builders return.
"""

from __future__ import annotations

from .api import (
    BENCHMARK_API_VERSION,
    COMMAND_METHODS,
    ENTRY_POINT_GROUP,
    PURE_METHODS,
    Benchmark,
    validate_plugin,
)

__all__ = [
    "BENCHMARK_API_VERSION",
    "COMMAND_METHODS",
    "ENTRY_POINT_GROUP",
    "PURE_METHODS",
    "Benchmark",
    "validate_plugin",
]
