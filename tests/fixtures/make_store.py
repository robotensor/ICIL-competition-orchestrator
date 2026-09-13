"""Write the fixture store: a small, byte-reproducible franka_1arm history for the dashboard.

    python tests/fixtures/make_store.py [OUT]      # default: tests/fixtures/store

It holds a genesis, one `light` duel in which a replay challenger dethrones a zero-action king (one
unit void), and a queue with one entry waiting. Units are derived through the fake benchmark exactly
as a duel derives them, so their ids, seeds and skills are what the orchestrator would publish.

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
from icil_orchestrator.ids import SubmissionRef, duel_id, event_id  # noqa: E402
from icil_orchestrator.queue import Queue  # noqa: E402
from icil_orchestrator.spec import load_spec_file  # noqa: E402
from icil_orchestrator.store.records import (  # noqa: E402
    duel_event,
    empty_skill_scores,
    index_record,
    unit_verdict_from_unit,
)
from icil_orchestrator.store.writer import Store  # noqa: E402

TRACK = "franka_1arm"
SIZE = "light"
#: Public on purpose: this key signs nothing that matters.
FIXTURE_SEED = hashlib.sha256(b"icil-orchestrator fixture store; not a secret").digest()

KING = SubmissionRef.make("robotensor/icil-zero-policy", "0" * 39 + "1")
CHALLENGER = SubmissionRef.make("robotensor/icil-replay-policy", "0" * 39 + "2")
WAITING = SubmissionRef.make("robotensor/icil-example-policy", "0" * 39 + "3")


def build(out: Path) -> Path:
    spec = load_spec_file(ROOT / "spec.json")
    if out.exists():
        shutil.rmtree(out)
    signer = Signer(FIXTURE_SEED)
    store = Store(out, spec, signer)
    store.init(signer.verify_key_hex)
    schema = int(spec.store["schema"])
    margin = spec.score_margin(TRACK)

    # -- genesis: the zero-action policy takes the empty throne
    genesis = index_record(
        schema=schema,
        event_id=event_id("genesis", TRACK, 1, KING.key),
        kind="genesis",
        track=TRACK,
        block=1,
        finished_at="2026-09-13T09:00:00Z",
        king=KING,
        challenger=None,
        king_scores=None,
        challenger_scores=None,
        score_margin=margin,
        dethroned=False,
        new_king=None,
    )
    _publish(store, spec, genesis, units=[], started_at="2026-09-13T09:00:00Z", wall=0.0)

    # -- one duel on the fake benchmark's units
    did = duel_id(spec.version, TRACK, CHALLENGER, KING)
    derived = plugin_units(
        spec, TRACK, did, SIZE, resolve=lambda name: icil_fake_benchmark.BENCHMARK
    )
    units, prompts = [], []
    for n, unit in enumerate(derived):
        verdict = unit_verdict_from_unit(unit, view=spec.demo_view(TRACK))
        prompt_sha = canonical_sha256({"unit_id": unit["unit_id"], "task": unit["task"]})
        verdict["prompt"]["sha256"] = prompt_sha
        prompts.append({"unit_id": unit["unit_id"], "sha256": prompt_sha})
        if n == 4:
            verdict.update(void=True, challenger_error="fake: unit exceeded its 600s budget")
        else:
            won = n % 3 != 2
            verdict.update(
                king_success=False,
                challenger_success=won,
                outcome="challenger" if won else "tie",
                king_steps=unit["instance_params"]["scene_seed"] % 300 + 100,
                challenger_steps=120 + n,
                king_progress=0.0,
                challenger_progress=1.0 if won else 0.5,
            )
        units.append(verdict)

    king_scores, challenger_scores = _scores(spec, units)
    tally = {
        "wins": sum(u["outcome"] == "challenger" and not u["void"] for u in units),
        "losses": sum(u["outcome"] == "king" and not u["void"] for u in units),
        "ties": sum(u["outcome"] == "tie" and not u["void"] for u in units),
        "void": sum(u["void"] for u in units),
    }
    tally["decided"] = tally["wins"] + tally["losses"]
    dethroned = challenger_scores["average"] >= king_scores["average"] + margin / 100
    duel = index_record(
        schema=schema,
        event_id=event_id("duel", TRACK, 2, did),
        kind="duel",
        track=TRACK,
        block=2,
        finished_at="2026-09-13T11:30:00Z",
        king=KING,
        challenger=CHALLENGER,
        king_scores=king_scores,
        challenger_scores=challenger_scores,
        score_margin=margin,
        dethroned=dethroned,
        new_king=CHALLENGER if dethroned else None,
        tally=tally,
        duel_size=SIZE,
        duel_id=did,
    )
    sides = {
        side: {"commit": ref.revision, "base_image_digest": None, "wall_seconds": 1800.0}
        for side, ref in (("king", KING), ("challenger", CHALLENGER))
    }
    _publish(
        store,
        spec,
        duel,
        units=units,
        started_at="2026-09-13T10:00:00Z",
        wall=5400.0,
        sides=sides,
        prompts=prompts,
        size=SIZE,
    )

    # -- the queue as the dashboard sees it
    queue = Queue(out.parent / f".{out.name}-queue.json")
    queue.state.block = 2
    queue.add(WAITING.repo, WAITING.revision, duel_size="smoke", now="2026-09-13T11:45:00Z")
    king = CHALLENGER if dethroned else KING
    snapshot = queue.snapshot(TRACK, king, schema, now="2026-09-13T12:00:00Z")
    store.write_queue(TRACK, snapshot)
    queue.path.unlink()
    (out / ".orchestrator.lock").unlink(missing_ok=True)
    return out


def _scores(spec, units):
    skills = spec.skills(TRACK)
    out = []
    for side in ("king", "challenger"):
        scores = empty_skill_scores(skills)
        for skill in skills:
            runs = [u[f"{side}_success"] for u in units if u["skill"] == skill and not u["void"]]
            scores[skill] = sum(runs) / len(runs) if runs else None
        rated = [scores[s] for s in skills if scores[s] is not None]
        scores["average"] = sum(rated) / len(rated) if rated else None
        out.append(scores)
    return out


def _publish(store, spec, record, *, units, started_at, wall, sides=None, prompts=None, size=None):
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
        notes=["Fixture store written by tests/fixtures/make_store.py; not a competition result."],
    )
    store.write_event(TRACK, event)
    store.append(TRACK, record)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "store"
    print(build(target.resolve()))
