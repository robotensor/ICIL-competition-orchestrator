"""`bpp-runtime`: check, convert, template, parity.

    bpp-runtime check    --weights DIR_OR_FILE [--template DIR] [--max-bytes N] [--no-hash]
    bpp-runtime convert  --ckpt X.ckpt --out DIR [--template DIR] [--overwrite]
    bpp-runtime template --ckpt X.ckpt --out DIR [--name NAME] [--exec-action-horizon 12]
    bpp-runtime parity   --ckpt X.ckpt --weights DIR --prompt PROMPT.npz [--seed 0] [--steps 30]

Each prints one JSON report on standard output and exits 0 when it passed, 1 when it did not and
2 on a usage error. Only `check` runs without the `model` extra, and it imports no torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import ARCHITECTURE
from .check import DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_HEADER_BYTES
from .template import DEFAULT_EXEC_ACTION_HORIZON

EXIT_OK = 0
EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bpp-runtime", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="check model.safetensors against the template")
    check.add_argument("--weights", required=True, help="model.safetensors, or its directory")
    check.add_argument("--template", help="the template directory (default: the packaged one)")
    check.add_argument("--architecture", default=ARCHITECTURE)
    check.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    check.add_argument("--max-header-bytes", type=int, default=DEFAULT_MAX_HEADER_BYTES)
    check.add_argument("--no-hash", action="store_true", help="skip weights_sha256")

    convert = commands.add_parser("convert", help="training checkpoint -> model.safetensors")
    convert.add_argument("--ckpt", required=True, help="a checkpoint you made: it is unpickled")
    convert.add_argument("--out", required=True, help="the directory to write model.safetensors to")
    convert.add_argument("--template")
    convert.add_argument("--architecture", default=ARCHITECTURE)
    convert.add_argument("--overwrite", action="store_true")

    template = commands.add_parser("template", help="write the architecture template")
    template.add_argument("--ckpt", required=True)
    template.add_argument("--out", required=True)
    template.add_argument("--name", default=ARCHITECTURE)
    template.add_argument("--exec-action-horizon", type=int, default=DEFAULT_EXEC_ACTION_HORIZON)

    parity = commands.add_parser("parity", help="converted weights act as the checkpoint does")
    parity.add_argument("--ckpt", required=True)
    parity.add_argument("--weights", required=True)
    parity.add_argument("--prompt", required=True, help="a benchmark prompt.npz")
    parity.add_argument("--seed", type=int, default=0)
    parity.add_argument("--steps", type=int, default=30)
    parity.add_argument("--device", default="cuda:0")
    parity.add_argument("--template")
    parity.add_argument("--tolerance", type=float, default=1e-5)
    parity.add_argument(
        "--xpolicylab", help="a directory holding the XPolicyLab package, for the HDF5 path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        from .check import check

        report = check(
            args.weights,
            template=args.template,
            architecture=args.architecture,
            max_file_bytes=args.max_bytes,
            max_header_bytes=args.max_header_bytes,
            compute_sha256=not args.no_hash,
        )
        return _emit(report.to_dict(), report.ok)
    if args.command == "convert":
        from .convert import ConvertError, convert

        try:
            report, info = convert(
                args.ckpt,
                args.out,
                template=args.template,
                architecture=args.architecture,
                overwrite=args.overwrite,
            )
        except ConvertError as exc:
            return _emit({"ok": False, "errors": [str(exc)]}, False)
        return _emit({**report.to_dict(), "conversion": info}, report.ok)
    if args.command == "template":
        from .template import generate

        summary = generate(
            args.ckpt, args.out, name=args.name, exec_action_horizon=args.exec_action_horizon
        )
        return _emit({"ok": True, **summary}, True)
    if args.command == "parity":
        from .parity import run

        report = run(
            args.ckpt,
            args.weights,
            args.prompt,
            seed=args.seed,
            steps=args.steps,
            device=args.device,
            xpolicylab=args.xpolicylab,
            tolerance=args.tolerance,
            template=args.template,
        )
        return _emit(report, report["ok"])
    raise AssertionError(args.command)


def _emit(report: dict[str, Any], ok: bool) -> int:
    json.dump(report, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return EXIT_OK if ok else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
