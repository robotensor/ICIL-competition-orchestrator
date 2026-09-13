"""The fake benchmark's command half: what a simulator subprocess would be.

Run as a script, never imported by the plugin. It imports `icil_fake_simulator` first, the way a
real command half imports SAPIEN, so the "simulator" is loaded here and only here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import icil_fake_simulator  # noqa: E402

CLIP = b"\x00\x00\x00\x18ftypmp42" + b"fake-clip" * 8


def materialize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = {"task": args.task, "scene_seed": args.scene_seed, "sim": icil_fake_simulator.NAME}
    (out / "prompt.npz").write_text(json.dumps(prompt, sort_keys=True))
    (out / "demonstration.mp4").write_bytes(CLIP)
    (out / "result.json").write_text(json.dumps({"success": True, "void": False, "steps": 5}))
    return 0


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
    result = {
        "success": behaviour == "succeed",
        "void": False,
        "steps": 5,
        "error": None,
        "progress": 1.0 if behaviour == "succeed" else 0.25,
        "prompt_sha256": icil_fake_simulator.digest(Path(args.prompt)),
    }
    (out / "result.json").write_text(json.dumps(result))
    (out / "evaluation.mp4").write_bytes(CLIP)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="icil-fake-benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--out", required=True)
    m.add_argument("--task", required=True)
    m.add_argument("--scene-seed", type=int, required=True)
    r = sub.add_parser("run")
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--policy-address", required=True)
    r.add_argument("--authkey-env", required=True)
    r.add_argument("--behaviour", default="succeed")
    args = parser.parse_args()
    return materialize(args) if args.cmd == "materialize" else run(args)


if __name__ == "__main__":
    sys.exit(main())
