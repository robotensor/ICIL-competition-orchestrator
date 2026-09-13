from icil_orchestrator.ids import (
    SubmissionRef,
    duel_id,
    event_id,
    is_repo,
    is_sha_revision,
    submission_key,
    unit_id,
    unit_seed,
)
from icil_orchestrator.rng import HashRng


def test_ids_are_stable():
    ref = SubmissionRef.make("owner/policy", "0123456789abcdef0123456789abcdef01234567")
    assert ref.key == submission_key("owner/policy", "0123456789abcdef0123456789abcdef01234567")
    assert len(ref.key) == 16
    king = SubmissionRef.make("org/king", "abcdef0123456789abcdef0123456789abcdef01")
    did = duel_id(7, "franka_1arm", ref, king)
    assert len(did) == 64 and did == duel_id(7, "franka_1arm", ref, king)
    assert did != duel_id(6, "franka_1arm", ref, king)
    assert did != duel_id(7, "franka_1arm", ref, None)
    assert unit_seed(did, "franka_stacking", 0) != unit_seed(did, "franka_stacking", 1)
    assert unit_seed(did, "franka_stacking", 0) != unit_seed(did, "franka_press_push", 0)
    assert 0 <= unit_seed(did, "franka_press_push", 3) < 2**32
    assert unit_id("fs", 7) == "fs-007"
    assert event_id("duel", "franka_1arm", 4, did) != event_id("duel", "franka_1arm", 5, did)
    assert SubmissionRef.from_dict(ref.as_dict()) == ref and SubmissionRef.from_dict(None) is None


def test_golden_values():
    """Frozen, and identical to the values icilval published: a change here changes every
    published id. CI runs this on Python 3.10 and 3.12."""
    assert submission_key("a/b", "c") == "fdd11077ff89f6bf"
    assert unit_seed("d", "pick_and_place", 0) == 2563984751
    challenger = SubmissionRef.make("org/challenger", "1" * 40)
    king = SubmissionRef.make("org/king", "2" * 40)
    did = duel_id(7, "franka_1arm", challenger, king)
    assert did == "7eab00ae0613e3785d960a714c09c17cfbdd39c8c0d60b06b707b833c8b57c46"
    assert unit_seed(did, "franka_stacking", 0) == 1670043122
    assert event_id("duel", "franka_1arm", 3, did) == (
        "ced309c94f9072bd4e3e09af5c0e463b2e7e67487adfb1b28dc684d7737a8128"
    )


def test_repo_and_revision_shapes():
    assert is_repo("org/name") and not is_repo("not a repo") and not is_repo("org/")
    assert is_sha_revision("abcdef0") and not is_sha_revision("main")


def test_hash_rng_determinism_and_range():
    a, b = HashRng("x", 1), HashRng("x", 1)
    assert [a.below(10) for _ in range(20)] == [b.below(10) for _ in range(20)]
    c = HashRng("x", 2)
    assert [a.below(10) for _ in range(20)] != [c.below(10) for _ in range(20)]
    r = HashRng("s")
    assert all(0 <= r.below(7) < 7 for _ in range(500))
    assert all(0.0 <= r.uniform() < 1.0 for _ in range(500))
    perm = HashRng("p").shuffled(list(range(50)))
    assert sorted(perm) == list(range(50)) and perm != list(range(50))
