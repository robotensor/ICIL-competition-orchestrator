"""The fake benchmark's command half: what a simulator subprocess would be.

Run as a script, never imported by the plugin. It imports `icil_fake_simulator` first, the way a
real command half imports SAPIEN, so the "simulator" is loaded here and only here.

`materialize` writes a prompt shaped like RoboTwin's: named arrays (`frames_head_camera`, `qpos`,
`actions`, ...) and a privileged `meta` array (the task and the scene seed as JSON bytes) that
never reaches a policy. The actions are a function of the scene seed.

`run` with `--behaviour policy` (what a duel's units get) drives the served policy through
`icil_policy.client.RemotePolicy`: hello, the demonstration without `meta`, reset, then one `act`
per demonstrated action. The episode succeeds iff the policy's actions are the demonstration's, so
the replay example wins and the zero example loses. A policy that is lost or does not answer in
time ends the episode void with `void_cause: "policy"`, as the RoboTwin plugin reports it: the
orchestrator counts that as the side's failure, not a void for both.

It keeps time as RoboTwin's run-unit does. With `--unit-timeout-s`, no call to the policy runs
within `RESULT_RESERVE_S` of that timeout, counted from the command's start, and a call cut short
there voids the unit on the harness, written before whoever started the command kills it.
`--step-s` is how long the "simulator" takes over each step, never past that deadline: a harness
that runs long.

`--behaviour policy_then_void` drives the policy the same way and then reports the unit void, with
`--void-cause` as its `void_cause` when one is given: a simulator that lost the scene after the
policy was done with it.

Every run appends a line to `runs.log` in its directory, so a test can count how often a unit ran.
Both commands write `given.json` there first: the arguments they were given and their environment,
the policy's key left out, so a test can see what the orchestrator passed a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

#: When the command started: a unit's timeout counts from here.
STARTED = time.monotonic()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import icil_fake_simulator  # noqa: E402

from icil_fake_benchmark import RESULT_RESERVE_S  # noqa: E402

CLIP = b"\x00\x00\x00\x18ftypmp42" + b"fake-clip" * 8
#: Frames in a demonstration; there is one action fewer.
STEPS = 6
ACTION_DIM = 16


def clip(tag: str) -> bytes:
    """A clip whose bytes say what it shows, so two different rollouts never share a sha."""
    return CLIP + hashlib.sha256(tag.encode()).hexdigest().encode()


def record_given(out: Path, args: argparse.Namespace) -> None:
    """`given.json`: the command's arguments and environment, without the policy's key."""
    secret = getattr(args, "authkey_env", None)
    environ = {k: v for k, v in os.environ.items() if k != secret}
    given = {"args": vars(args), "environ": environ}
    (out / "given.json").write_text(json.dumps(given, sort_keys=True))


def materialize(args: argparse.Namespace) -> int:
    import numpy as np

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record_given(out, args)
    behaviour = args.behaviour
    if behaviour == "crash":
        print("the expert lost the GPU", file=sys.stderr)
        return 3
    if behaviour == "hang":
        time.sleep(600)
        return 0
    rng = np.random.default_rng(args.scene_seed)
    seed = args.scene_seed + (1 if behaviour == "wrong" else 0)
    meta = {"task": args.task, "scene_seed": seed, "sim": icil_fake_simulator.NAME}
    arrays = {
        "frames_head_camera": rng.integers(0, 255, (STEPS, 4, 4, 3), dtype=np.uint8),
        "qpos": rng.standard_normal((STEPS, ACTION_DIM)),
        "actions": rng.standard_normal((STEPS - 1, ACTION_DIM)),
        "frequency": np.array(10.0),
        "meta": np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
    }
    with open(out / "prompt.npz", "wb") as fh:
        np.savez(fh, **arrays)
    (out / "demonstration.mp4").write_bytes(clip(f"demo|{args.task}|{args.scene_seed}"))
    expert = behaviour != "expert_fails"
    result = {"success": expert, "void": False, "steps": STEPS - 1}
    if not expert:
        result["error"] = "the expert never succeeded"
    (out / "result.json").write_text(json.dumps(result))
    return 0


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.log", "a") as fh:
        fh.write(f"{os.getpid()}\n")
    record_given(out, args)
    if args.authkey_env not in os.environ:
        print(f"no policy authkey in ${args.authkey_env}", file=sys.stderr)
        return 4
    behaviour = args.behaviour
    if behaviour == "crash":
        print("the simulator lost the GPU", file=sys.stderr)
        return 3
    if behaviour == "hang":
        time.sleep(600)
        return 0
    if behaviour == "silent":
        return 0
    if behaviour == "garbage":
        (out / "result.json").write_text("{not json")
        return 0
    prompt_sha256 = icil_fake_simulator.digest(Path(args.prompt))
    if behaviour == "policy":
        result = drive_policy(args)
    elif behaviour == "policy_then_void":
        drive_policy(args)
        result = {"success": None, "void": True, "steps": None, "error": "the simulator lost it"}
        if args.void_cause:
            result["void_cause"] = args.void_cause
    else:
        result = {
            "success": behaviour == "succeed",
            "void": False,
            "steps": 5,
            "error": None,
            "progress": 1.0 if behaviour == "succeed" else 0.25,
        }
    result["prompt_sha256"] = prompt_sha256
    (out / "result.json").write_text(json.dumps(result))
    tag = f"eval|{prompt_sha256}|{result['success']}|{result['steps']}|{result.get('error')}"
    (out / "evaluation.mp4").write_bytes(clip(tag))
    return 0


class OutOfTime(Exception):
    """A call to the policy the unit's deadline cut short: the harness's time ran out."""

    def __init__(self, op: str) -> None:
        super().__init__(f"the unit ran out of time during {op}")


def drive_policy(args: argparse.Namespace) -> dict:
    """One episode against the served policy: act once per demonstrated action."""
    import numpy as np

    from icil_policy.client import PolicyUnavailable, RemotePolicy

    deadline = math.inf
    if args.unit_timeout_s is not None:
        deadline = STARTED + args.unit_timeout_s - RESULT_RESERVE_S
    with np.load(args.prompt) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(bytes(arrays.pop("meta")).decode())  # privileged: stays on this side
    demonstrated = arrays["actions"]
    taken: list = []
    error = cause = None
    try:
        key = bytes.fromhex(os.environ[args.authkey_env])
        with RemotePolicy(args.policy_address, key, timeout_s=args.act_timeout_s) as policy:

            def call(op, method, *call_args):
                limit = min(args.act_timeout_s, deadline - time.monotonic())
                if limit <= 0:
                    raise OutOfTime(op)
                policy.timeout_s = limit
                began = time.monotonic()
                try:
                    return method(*call_args)
                except PolicyUnavailable:
                    cut = limit < args.act_timeout_s and time.monotonic() - began >= limit
                    if cut:
                        raise OutOfTime(op) from None
                    raise

            call("hello", policy.hello)
            info = {"frequency": 10.0, "cameras": ["head_camera"]}
            call("prompt", policy.set_demonstration, arrays, info)
            call("reset", policy.reset, int(meta["scene_seed"]))
            for t in range(len(demonstrated)):
                action = call("act", policy.act, {"qpos": arrays["qpos"][t]})["action"]
                taken.append(np.atleast_2d(np.asarray(action, dtype=np.float64))[0])
                time.sleep(max(0.0, min(args.step_s, deadline - time.monotonic())))
    except OutOfTime as exc:
        error, cause = str(exc), "harness"
    except PolicyUnavailable as exc:
        error, cause = f"policy: {exc}", "policy"
    if error is not None:
        print(error, file=sys.stderr)
    matched = sum(
        1
        for t, action in enumerate(taken)
        if action.shape == demonstrated[t].shape and np.allclose(action, demonstrated[t])
    )
    if error is not None:
        return {
            "success": None,
            "void": True,
            "void_cause": cause,
            "steps": len(taken),
            "error": error,
            "progress": matched / len(demonstrated),
        }
    return {
        "success": matched == len(demonstrated),
        "void": False,
        "steps": len(taken),
        "error": None,
        "progress": matched / len(demonstrated),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="icil-fake-benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--out", required=True)
    m.add_argument("--task", required=True)
    m.add_argument("--scene-seed", type=int, required=True)
    m.add_argument("--behaviour", default="succeed")
    r = sub.add_parser("run")
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--policy-address", required=True)
    r.add_argument("--authkey-env", required=True)
    r.add_argument("--behaviour", default="policy")
    r.add_argument("--act-timeout-s", type=float, default=30.0)
    r.add_argument("--unit-timeout-s", type=float, default=None)
    r.add_argument("--step-s", type=float, default=0.0)
    r.add_argument("--void-cause", default="")
    args = parser.parse_args()
    return materialize(args) if args.cmd == "materialize" else run(args)


if __name__ == "__main__":
    sys.exit(main())
