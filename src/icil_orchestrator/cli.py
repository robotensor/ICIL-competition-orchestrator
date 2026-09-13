"""`icil-orchestrator` command line. argparse only, and every import is local to its command, so
`--help` and a listing work on a host with nothing else installed."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def _spec(args: argparse.Namespace):
    from .spec import load_spec

    return load_spec(args.spec)


def cmd_benchmarks(args: argparse.Namespace) -> int:
    spec = _spec(args)
    if args.benchmarks_cmd == "list":
        from .benchmarks.plugins import discover

        rows = discover(spec)
        if args.json:
            print(json.dumps([r.as_dict() for r in rows.values()], indent=2, sort_keys=True))
            return 0
        for row in rows.values():
            where = f"{row.distribution or '-'} {row.version or ''}".strip()
            state = "; ".join(row.problems) if row.problems else "ok"
            notes = f" ({'; '.join(row.notes)})" if row.notes else ""
            print(f"{row.name:<12} {where:<36} {state}{notes}")
        return 0

    from .benchmarks.check import check_benchmark

    report = check_benchmark(spec, args.id)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        for problem in report.problems:
            print(f"problem: {problem}")
        for note in report.notes:
            print(f"note: {note}")
        print(f"{report.name}: {'ok' if report.ok else 'REFUSED'}")
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="icil-orchestrator", description="ICIL competition orchestrator"
    )
    p.add_argument("--spec", help="path to spec.json (default: packaged, or the repository's)")
    sub = p.add_subparsers(dest="cmd", required=True)

    bm = sub.add_parser("benchmarks", help="the benchmarks plugged into this host")
    bm_sub = bm.add_subparsers(dest="benchmarks_cmd", required=True)
    bm_list = bm_sub.add_parser("list", help="every declared or installed benchmark (no import)")
    bm_list.add_argument("--json", action="store_true")
    bm_check = bm_sub.add_parser("check", help="load one benchmark and check it against the spec")
    bm_check.add_argument("id")
    bm_check.add_argument("--json", action="store_true")
    bm.set_defaults(func=cmd_benchmarks)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
