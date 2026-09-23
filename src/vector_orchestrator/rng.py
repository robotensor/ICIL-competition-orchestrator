"""A tiny deterministic generator built on sha256.

Anything published that looks random - a draw, a shuffle - must be reproducible by anyone from the
published ids, in any language and on any Python version. Python's `random` module does not promise
that across versions; a hash counter does.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")

_MAX = 1 << 64


class HashRng:
    def __init__(self, *parts: Any):
        self._seed = "|".join(str(p) for p in parts).encode("utf-8")
        self._counter = 0

    def _next(self) -> int:
        self._counter += 1
        digest = hashlib.sha256(self._seed + b"#" + str(self._counter).encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def below(self, n: int) -> int:
        """Uniform integer in [0, n). Rejection sampling keeps it unbiased."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = _MAX - (_MAX % n)
        while True:
            x = self._next()
            if x < limit:
                return x % n

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * (self._next() / _MAX)

    def chance(self, p: float) -> bool:
        return self.uniform() < p

    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("empty sequence")
        return seq[self.below(len(seq))]

    def shuffled(self, seq: Sequence[T]) -> list[T]:
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = self.below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out
