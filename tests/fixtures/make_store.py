"""Write the fixture store: a small, byte-reproducible franka_1arm history for the dashboard.

    python tests/fixtures/make_store.py [OUT]      # default: tests/fixtures/store

It holds a genesis, one `light` duel in which a replay challenger dethrones a zero-action king (one
unit void for the harness), and a queue with one entry waiting, in the shapes the orchestrator
publishes: a genesis names its duel id and carries its units and scores, every unit its
`prompt_sha256`, and every event its `sides`, `scoring`, `benchmarks` and `runtime`. Units are
derived through the fake benchmark exactly as a duel derives them, so their ids, seeds and skills
are what the orchestrator would publish.

It is NOT a competition result. It is signed with a key derived from a public string (below), so
anyone can forge records under it; its only use is rendering the dashboard against the layout this
orchestrator writes (`ICIL_STORE=<abs path to tests/fixtures/store> npm run dev`) and catching an
unintended change to that layout. Clips are not included (units carry no video hashes).

After an intended change to spec.json or the store layout, regenerate it and commit the result.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fake_benchmark"))
sys.path.insert(0, str(ROOT / "src"))

import icil_fake_benchmark  # noqa: E402

from icil_orchestrator.benchmarks.units import plugin_units  # noqa: E402
from icil_orchestrator.canon import Signer, canonical_sha256  # noqa: E402
from icil_orchestrator.duel import score  # noqa: E402
from icil_orchestrator.ids import SubmissionRef, duel_id, event_id  # noqa: E402
from icil_orchestrator.queue import Queue  # noqa: E402
from icil_orchestrator.spec import load_spec_file  # noqa: E402
from icil_orchestrator.store.records import (  # noqa: E402
    duel_event,
    index_record,
    unit_tally,
    unit_verdict_from_unit,
)
from icil_orchestrator.store.writer import Store  # noqa: E402

TRACK = "franka_1arm"
SIZE = "light"
GENESIS_SIZE = "smoke"
#: Public on purpose: this key signs nothing that matters.
FIXTURE_SEED = hashlib.sha256(b"icil-orchestrator fixture store; not a secret").digest()

KING = SubmissionRef.make("robotensor/icil-zero-policy", "0" * 39 + "1")
CHALLENGER = SubmissionRef.make("robotensor/icil-replay-policy", "0" * 39 + "2")
WAITING = SubmissionRef.make("robotensor/icil-example-policy", "0" * 39 + "3")
BASE_DIGEST = "sha256:" + "b" * 64


def build(out: Path) -> Path:
    spec = load_spec_file(ROOT / "spec.json")
    if out.exists():
        shutil.rmtree(out)
    signer = Signer(FIXTURE_SEED)
    store = Store(out, spec, signer)
    store.init(signer.verify_key_hex)
    schema = int(spec.store["schema"])
    margin = spec.score_margin(TRACK)
    skills = spec.skills(TRACK)

    # -- genesis: the zero-action policy takes the empty throne, scored on its own units
    gid = duel_id(spec.version, TRACK, KING, None)
    units, prompts = _units(spec, gid, GENESIS_SIZE)
    for n, unit in enumerate(units):
        unit.update(king_success=False, king_steps=100 + n, king_progress=0.0, outcome="tie")
    king_scores = score.skill_scores(units, "king", skills)
    genesis = index_record(
        schema=schema,
        event_id=event_id("genesis", TRACK, 1, gid),
        kind="genesis",
        track=TRACK,
        block=1,
        finished_at="2026-09-13T09:00:00Z",
        king=KING,
        challenger=None,
        king_scores=king_scores,
        challenger_scores=None,
        score_margin=margin,
        dethroned=False,
        new_king=None,
        tally=unit_tally(units),
        duel_size=GENESIS_SIZE,
        duel_id=gid,
    )
    _publish(
        store,
        spec,
        genesis,
        units=units,
        prompts=prompts,
        size=GENESIS_SIZE,
        started_at="2026-09-13T08:30:00Z",
        wall=1800.0,
        sides={"king": _side(KING, units, "king", wall=1500.0)},
        scoring={"reason": "genesis", "score_margin": margin, "delta_points": None},
        notes=["The first entrant took the empty throne, scored on its own units."],
    )

    # -- one duel on the fake benchmark's units
    did = duel_id(spec.version, TRACK, CHALLENGER, KING)
    units, prompts = _units(spec, did, SIZE)
    for n, unit in enumerate(units):
        if n == 4:
            unit.update(void=True, challenger_error="fake: the simulator lost the scene")
        else:
            won = n % 3 != 2
            unit.update(
                king_success=False,
                challenger_success=won,
                outcome="challenger" if won else "tie",
                king_steps=unit["instance_params"]["scene_seed"] % 300 + 100,
                challenger_steps=120 + n,
                king_progress=0.0,
                challenger_progress=1.0 if won else 0.5,
            )
    verdict = score.verdict(units, margin, skills)
    duel = index_record(
        schema=schema,
        event_id=event_id("duel", TRACK, 2, did),
        kind="duel",
        track=TRACK,
        block=2,
        finished_at="2026-09-13T11:30:00Z",
        king=KING,
        challenger=CHALLENGER,
        king_scores=verdict.king_scores,
        challenger_scores=verdict.challenger_scores,
        score_margin=margin,
        dethroned=verdict.dethroned,
        new_king=CHALLENGER if verdict.dethroned else None,
        tally=unit_tally(units),
        duel_size=SIZE,
        duel_id=did,
    )
    _publish(
        store,
        spec,
        duel,
        units=units,
        prompts=prompts,
        size=SIZE,
        started_at="2026-09-13T10:00:00Z",
        wall=5400.0,
        sides={
            "challenger": _side(CHALLENGER, units, "challenger", wall=1800.0),
            "king": _side(KING, units, "king", wall=1800.0),
        },
        scoring=verdict.as_dict(),
        notes=[
            f"Crown rule: {verdict.reason} ({verdict.delta_points:+.2f} points against a margin "
            f"of {margin:g})."
        ],
    )

    # -- the queue as the dashboard sees it
    queue = Queue(out.parent / f".{out.name}-queue.json")
    queue.set_block(2)
    queue.add(WAITING.repo, WAITING.revision, duel_size="smoke", now="2026-09-13T11:45:00Z")
    king = CHALLENGER if verdict.dethroned else KING
    snapshot = queue.snapshot(TRACK, king, schema, now="2026-09-13T12:00:00Z")
    store.write_queue(TRACK, snapshot)
    queue.path.unlink()
    queue.path.with_name(queue.path.name + ".lock").unlink(missing_ok=True)
    # The store's own bookkeeping (locks) is never published and never part of the fixture.
    for dotfile in out.rglob(".*"):
        if dotfile.is_file():
            dotfile.unlink()
    return out


def _units(spec, did: str, size: str) -> tuple[list[dict], list[dict]]:
    """A duel's published unit rows, with a stand-in prompt hash each, and its `prompts`."""
    derived = plugin_units(
        spec, TRACK, did, size, resolve=lambda name: icil_fake_benchmark.BENCHMARK
    )
    units, prompts = [], []
    for unit in derived:
        row = unit_verdict_from_unit(unit, view=spec.demo_view(TRACK))
        prompt_sha = canonical_sha256({"unit_id": unit["unit_id"], "task": unit["task"]})
        row["prompt"]["sha256"] = row["prompt_sha256"] = prompt_sha
        units.append(row)
        prompts.append({"unit_id": unit["unit_id"], "sha256": prompt_sha, "demo_video": None})
    return units, prompts


def _side(ref: SubmissionRef, units: list[dict], side: str, *, wall: float) -> dict:
    """What an event records of one side, as the docker runtime prepares it."""
    return {
        **ref.as_dict(),
        "commit": ref.revision,
        "base_image_digest": BASE_DIGEST,
        "image": "sha256:" + hashlib.sha256(ref.key.encode()).hexdigest(),
        "policy": "replay.policy:ReplayPolicy" if ref == CHALLENGER else "zero.policy:ZeroPolicy",
        "action_type": "qpos",
        "runtime": "docker",
        "refused": None,
        "wall_seconds": wall,
        "units": len(units),
        "void": sum(1 for u in units if u["void"] and u.get(f"{side}_error")),
    }


def _publish(store, spec, record, *, units, prompts, size, started_at, wall, sides, scoring, notes):
    event = duel_event(
        record,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        units=units,
        units_per_skill=spec.units_per_skill(TRACK, size),
        started_at=started_at,
        wall_seconds=wall,
        sides=sides,
        demonstration=spec.demonstration(TRACK),
        prompts=prompts,
        notes=[*notes, "Fixture store written by tests/fixtures/make_store.py; not a result."],
    )
    event["runtime"] = "docker"
    event["scoring"] = {
        **scoring,
        "void_fraction": score.void_fraction(units),
        "max_void_fraction": spec.max_void_fraction(TRACK),
    }
    event["benchmarks"] = {
        name: {"info": {"id": name}, "pin": spec.benchmark_pin(name), "installed_version": None}
        for name in spec.benchmarks_of(TRACK)
    }
    store.write_event(TRACK, event)
    store.append(TRACK, record)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "store"
    print(build(target.resolve()))
