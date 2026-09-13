"""`icil-orchestrator` command line. argparse only, and every import is local to its command, so
`--help` and a listing work on a host with nothing else installed."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

#: Where `store init` keeps the signing key unless told otherwise. Never inside the store: the
#: store is mirrored, and a key in it would be published.
DEFAULT_KEY = "keys/orchestrator.ed25519"


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


def cmd_store(args: argparse.Namespace) -> int:
    spec = _spec(args)
    if args.store_cmd == "init":
        from .canon import Signer
        from .store.writer import Store, store_lock

        key = Path(args.key)
        root = Path(args.root).resolve()
        if key.resolve().is_relative_to(root):
            print(f"error: the signing key must live outside the store ({root})", file=sys.stderr)
            return 2
        if key.exists():
            signer = Signer.from_file(key)
        else:
            signer = Signer.generate()
            signer.save(key)
            print(f"generated signing key {key} (mode 0600)", file=sys.stderr)
        store = Store(root, spec, signer)
        existing = store.manifest()
        if existing and existing.get("validator_key") != signer.verify_key_hex:
            print(
                f"error: {root} is signed by {existing.get('validator_key')}, not by {key}; "
                "re-initialising it with another key would invalidate every record",
                file=sys.stderr,
            )
            return 1
        with store_lock(root):
            manifest = store.init(signer.verify_key_hex)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.store_cmd == "verify":
        from .store.verify import verify_store

        report = verify_store(args.root, spec, validator_key=args.validator_key)
        print(f"validator_key: {report.validator_key}")
        for w in report.warnings:
            print("warning:", w)
        for e in report.errors:
            print("error:", e)
        state = "OK" if report.ok else "FAILED"
        print(f"records={report.records} events={report.events} media={report.media} {state}")
        return 0 if report.ok else 1

    from .store.mirror import mirror_store

    n = mirror_store(
        args.root, args.repo, message=args.message, all_files=args.all, prune=args.prune
    )
    print(f"mirrored {n} files to {args.repo}")
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    from .ids import SubmissionRef, is_repo
    from .queue import Queues

    spec = _spec(args)
    try:
        track = args.track or spec.sole_track
        queue = Queues(args.queue, spec.tracks)[track]
    except (KeyError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2

    if args.queue_cmd == "add":
        if not is_repo(args.repo):
            print(
                f"error: {args.repo!r} is not a Hugging Face repo id (owner/name)", file=sys.stderr
            )
            return 2
        if args.duel_size is not None and args.duel_size not in spec.sizes(track):
            sizes = ", ".join(spec.sizes(track))
            print(f"error: duel size {args.duel_size!r} is not one of {sizes}", file=sys.stderr)
            return 2
        entry, position = queue.add(
            args.repo, args.revision, duel_size=args.duel_size, source="cli"
        )
        print(f"{entry.ref.entry} key={entry.key} track={track} position={position}")
    elif args.queue_cmd == "list":
        for i, e in enumerate(queue.entries(), start=1):
            size = e.duel_size or "-"
            print(f"{i:3d} {e.key} {e.repo}@{e.revision} size={size} accepted={e.accepted_at}")
        ip = queue.state.in_progress
        print(f"in_progress={ip.event_id if ip else '-'} block={queue.block}")
    else:
        if not queue.remove(args.key):
            print(f"not found: {args.key}")
            return 1
        print(f"removed {args.key}")

    if args.store and args.queue_cmd != "list":
        from .store.writer import Store, store_lock

        store = Store(args.store, spec)
        try:
            with store_lock(store.root):
                head = store.head(track) or {}
                king = SubmissionRef.from_dict(head.get("king"))
                store.write_queue(track, queue.snapshot(track, king, int(spec.store["schema"])))
        except RuntimeError as exc:
            # The queue itself is changed; only its published snapshot is left to the orchestrator
            # that holds the store, which rewrites it every cycle anyway.
            print(f"note: {exc}; it will publish the queue snapshot", file=sys.stderr)
    return 0


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

    st = sub.add_parser("store", help="the signed result store")
    st_sub = st.add_subparsers(dest="store_cmd", required=True)
    st_init = st_sub.add_parser("init", help="write the manifest and an empty head per track")
    st_init.add_argument("root")
    st_init.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"ed25519 seed file, generated (0600) if absent (default: {DEFAULT_KEY})",
    )
    st_verify = st_sub.add_parser(
        "verify", help="check signatures, sequence, events, media, schema"
    )
    st_verify.add_argument("root")
    st_verify.add_argument(
        "--validator-key",
        default=None,
        help="the ed25519 public key (hex) this store must be signed by; without it the store's "
        "own unsigned manifest says which key to trust",
    )
    st_mirror = st_sub.add_parser("mirror", help="push the store to a Hugging Face dataset repo")
    st_mirror.add_argument("root")
    st_mirror.add_argument("--repo", required=True, help="Hugging Face dataset repo owner/name")
    st_mirror.add_argument("--message", default="publish")
    st_mirror.add_argument("--all", action="store_true", help="upload every file")
    st_mirror.add_argument(
        "--prune",
        action="store_true",
        help="make the repo exactly this store in one commit (for a rebuilt store)",
    )
    st.set_defaults(func=cmd_store)

    q = sub.add_parser("queue", help="the challenger queue, one per track")
    q.add_argument("--queue", default="queue", help="queue directory, one file per track")
    q.add_argument("--track", default=None, help="which track (default: the only one)")
    q.add_argument(
        "--store", default=None, help="also publish tracks/{track}/queue.json to this store"
    )
    q_sub = q.add_subparsers(dest="queue_cmd", required=True)
    q_add = q_sub.add_parser("add", help="queue repo@revision at the back")
    q_add.add_argument("repo")
    q_add.add_argument("revision")
    q_add.add_argument("--duel-size", default=None)
    q_sub.add_parser("list")
    q_rm = q_sub.add_parser("remove", help="drop an entry by its key")
    q_rm.add_argument("key")
    q.set_defaults(func=cmd_queue)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
