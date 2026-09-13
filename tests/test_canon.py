from icil_orchestrator.canon import (
    Signer,
    canonical_json,
    canonical_sha256,
    sha256_hex,
    verify_signature,
)


def test_canonical_json_is_sorted_and_compact():
    assert (
        canonical_json({"b": 1, "a": [1, 2, {"z": None, "y": "é"}]})
        == '{"a":[1,2,{"y":"\\u00e9","z":null}],"b":1}'
    )


def test_sha256_matches_known_vector():
    assert sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert canonical_sha256({"a": 1}) == sha256_hex('{"a":1}')


def test_sign_verify_roundtrip_and_tamper(tmp_path):
    signer = Signer.generate()
    signer.save(tmp_path / "k")
    assert (tmp_path / "k").stat().st_mode & 0o777 == 0o600
    reloaded = Signer.from_file(tmp_path / "k")
    assert reloaded.verify_key_hex == signer.verify_key_hex
    msg = canonical_json({"seq": 1})
    sig = signer.sign(msg)
    assert verify_signature(signer.verify_key_hex, msg, sig)
    assert not verify_signature(signer.verify_key_hex, msg.replace("1", "2"), sig)
    assert not verify_signature(Signer.generate().verify_key_hex, msg, sig)
    assert not verify_signature(signer.verify_key_hex, msg, "zz")


def test_signatures_are_deterministic():
    """ed25519 signs deterministically, which is what lets a fixture store be byte-reproducible."""
    signer = Signer(bytes(range(32)))
    assert signer.sign("x") == Signer(bytes(range(32))).sign("x")
