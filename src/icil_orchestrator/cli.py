"""`icil-orchestrator` command line. argparse only, and every import is local to its command, so
`--help` and a listing work on a host with nothing else installed."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
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
        store = Store(root, spec)
        existing = store.manifest()
        if key.exists():
            signer = Signer.from_file(key)
        elif existing:
            # Generating one here would leave an unrelated seed behind for a later run to pick up
            # as the store's key.
            print(
                f"error: {root} is signed by {existing.get('validator_key')} and {key} does not "
                "exist, so it cannot be re-initialised; point --key at its signing key",
                file=sys.stderr,
            )
            return 1
        else:
            signer = Signer.generate()
            signer.save(key)
            print(f"generated signing key {key} (mode 0600)", file=sys.stderr)
        store.signer = signer
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

    try:
        n = mirror_store(args.root, args.repo, message=args.message, prune=args.prune)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"mirrored {n} files to {args.repo}")
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    from .ids import SubmissionRef, is_commit_sha, is_repo
    from .queue import Queues

    spec = _spec(args)
    try:
        track = args.track or spec.sole_track
        queue = Queues(args.queue, spec.tracks)[track]
    except (KeyError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2

    if args.queue_cmd == "add":
        if args.duel_size is not None and args.duel_size not in spec.sizes(track):
            sizes = ", ".join(spec.sizes(track))
            print(f"error: duel size {args.duel_size!r} is not one of {sizes}", file=sys.stderr)
            return 2
        revision = args.revision
        if is_repo(args.repo):
            # The Hub is asked once, here: a branch or a tag becomes the commit it names, a sha
            # is confirmed to be a commit of the repository, and the queue holds the commit.
            from .submissions import SubmissionError, SubmissionRejected
            from .submissions.resolve import resolve

            try:
                revision = resolve(args.repo, revision).sha
            except (SubmissionRejected, SubmissionError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            if not is_commit_sha(args.revision):
                print(f"resolved {args.repo}@{args.revision} to {revision}", file=sys.stderr)
        try:
            entry, position = queue.add(args.repo, revision, duel_size=args.duel_size, source="cli")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
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


def cmd_submission(args: argparse.Namespace) -> int:
    from .submissions.docker import Docker

    spec = _spec(args)
    docker = Docker()
    if args.submission_cmd == "build-base":
        from .submissions import SubmissionError
        from .submissions.image import build_base_image

        try:
            built = build_base_image(docker, spec, Path(args.context).resolve())
        except SubmissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(built.__dict__ | {"base": built.base.__dict__}, indent=2))
        else:
            print(f"built {built.tag} in {built.seconds:g}s", file=sys.stderr)
            print(built.image_id)
        return 0

    if args.submission_cmd == "prune":
        from .submissions import SubmissionError
        from .submissions.image import prune_submission_images

        try:
            removed, kept = prune_submission_images(docker)
        except SubmissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({"removed": removed, "kept": kept}, indent=2, sort_keys=True))
        else:
            for ref in removed:
                print(f"removed {ref}")
            for ref, why in kept.items():
                print(f"kept    {ref}: {why}")
        return 0

    from .submissions.check import check_submission
    from .submissions.fetch import HubFetcher, LocalFetcher, RepoCache

    repo, _, revision = args.ref.partition("@")
    if not repo or not revision:
        print(f"error: {args.ref!r} is not owner/name@revision", file=sys.stderr)
        return 2
    cache = RepoCache(args.cache, int(spec.submission["max_repo_bytes"]))
    fetcher = LocalFetcher(cache, args.local) if args.local else HubFetcher(cache)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="icil-check-"))
    try:
        report = check_submission(
            spec,
            repo,
            revision,
            fetcher=fetcher,
            docker=docker,
            work_dir=work,
            base_digest=args.base_image,
            gpus=args.gpus,
            build_timeout_s=args.build_timeout,
        )
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        for step in report.steps:
            detail = step.detail.replace("\n", "\n" + " " * 22)
            print(f"{step.name:<9} {step.status:<9} {step.seconds:7.1f}s  {detail}".rstrip())
        failed = report.failed_step
        where = f" at {failed.name}" if failed else ""
        print(f"{repo}@{report.sha or revision}: {report.verdict.upper()}{where}")
    return {"accepted": 0, "rejected": 1}.get(report.verdict, 2)


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
    q_add.add_argument(
        "revision", help="a branch, a tag or a commit sha; the queue holds the Hub's commit"
    )
    q_add.add_argument("--duel-size", default=None)
    q_sub.add_parser("list")
    q_rm = q_sub.add_parser("remove", help="drop an entry by its key")
    q_rm.add_argument("key")
    q.set_defaults(func=cmd_queue)

    sm = sub.add_parser("submission", help="a submission's image and sandbox")
    sm_sub = sm.add_subparsers(dest="submission_cmd", required=True)
    sm_check = sm_sub.add_parser(
        "check", help="resolve, fetch, check, build and run repo@revision, and say hello"
    )
    sm_check.add_argument("ref", help="owner/name@revision (a branch, a tag or a commit sha)")
    sm_check.add_argument(
        "--local", default=None, metavar="DIR", help="this directory in place of the Hub"
    )
    sm_check.add_argument(
        "--cache", default="cache", help="where checkouts are kept, by commit (default: cache)"
    )
    sm_check.add_argument(
        "--base-image",
        default=None,
        metavar="DIGEST",
        help="the base image digest while spec.json's is null (see `submission build-base`)",
    )
    sm_check.add_argument(
        "--work", default=None, metavar="DIR", help="keep the socket directory and log here"
    )
    sm_check.add_argument(
        "--gpus", type=int, default=None, help="GPUs for the container (default: the spec's)"
    )
    sm_check.add_argument(
        "--build-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="how long the image build (the requirements install) may take before the "
        "submission is rejected at build (default: submissions.check.BUILD_TIMEOUT_S)",
    )
    sm_check.add_argument("--json", action="store_true")
    sm_base = sm_sub.add_parser("build-base", help="build docker/policy-base and print its digest")
    sm_base.add_argument(
        "--context", default=".", help="the repository root (default: the current directory)"
    )
    sm_base.add_argument("--json", action="store_true")
    sm_prune = sm_sub.add_parser(
        "prune", help="remove the submission images no container was made from"
    )
    sm_prune.add_argument("--json", action="store_true")
    sm.set_defaults(func=cmd_submission)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
