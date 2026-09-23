"""Canonical JSON, hashing and ed25519 signatures.

Every published record is signed over its canonical JSON form, so the encoding is part of the
contract: sorted keys, no whitespace, ASCII only, no NaN. A third party checks the signature
against **the bytes of the line as published**, with the orchestrator's public key from
`manifest.json` - never against a re-encoding of the parsed record. Numbers are formatted as Python
formats them (`3.0` stays `3.0`, where `JSON.stringify` and RFC 8785 both give `3`), so a verifier
in another language that re-encoded first would reject valid lines. `store verify` re-encodes only
to check that this writer wrote the canonical form it signed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


class Signer:
    """ed25519 signer around a 32-byte seed. The verify key (hex) is the orchestrator's public id,
    published as `validator_key` in the store's manifest and in every live frame."""

    def __init__(self, seed: bytes):
        from nacl.signing import SigningKey

        if len(seed) != 32:
            raise ValueError("ed25519 seed must be 32 bytes")
        self._key = SigningKey(seed)
        self.verify_key_hex: str = self._key.verify_key.encode().hex()

    @classmethod
    def generate(cls) -> Signer:
        return cls(os.urandom(32))

    @classmethod
    def from_hex(cls, seed_hex: str) -> Signer:
        return cls(bytes.fromhex(seed_hex.strip()))

    @classmethod
    def from_file(cls, path: str | Path) -> Signer:
        return cls.from_hex(Path(path).read_text().strip())

    def seed_hex(self) -> str:
        return self._key.encode().hex()

    def save(self, path: str | Path) -> None:
        """Write the seed with mode 0600: it is the only thing that can publish as this store."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(self.seed_hex() + "\n")

    def sign(self, message: bytes | str) -> str:
        if isinstance(message, str):
            message = message.encode("utf-8")
        return self._key.sign(message).signature.hex()


def verify_signature(verify_key_hex: str, message: bytes | str, signature_hex: str) -> bool:
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    if isinstance(message, str):
        message = message.encode("utf-8")
    try:
        VerifyKey(bytes.fromhex(verify_key_hex)).verify(message, bytes.fromhex(signature_hex))
        return True
    except (BadSignatureError, ValueError):
        return False
