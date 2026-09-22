"""`icil-orchestrator` command line. argparse only, and every import is local to its command, so
`--help` and a listing work on a host with nothing else installed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
            # is confirmed to be a commit a branch or a tag of the repository holds - not a pull
            # request's alone - and the queue holds the commit.
            from .submissions import SubmissionError, SubmissionRejected
            from .submissions.resolve import resolve_for_queue

            try:
                revision = resolve_for_queue(args.repo, revision).sha
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


def cmd_admin(args: argparse.Namespace) -> int:
    from .admin import serve

    return serve(
        _spec(args),
        store_dir=args.store,
        queue_dir=args.queue,
        host=args.host,
        port=args.port,
        token_env=args.token_env,
    )


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


def _local_dirs(pairs: Sequence[str] | None) -> dict[str, str]:
    """`--local REPO=DIR` pairs, or `ValueError` naming the one that is not."""
    out: dict[str, str] = {}
    for pair in pairs or ():
        repo, sep, directory = pair.partition("=")
        if not sep or not repo or not directory:
            raise ValueError(f"--local {pair!r} is not REPO=DIR")
        out[repo] = directory
    return out


def _orchestrator(args: argparse.Namespace, spec):
    """The store, runtime, live reporter and mirror a duel or the daemon publishes with."""
    import os

    from .canon import Signer
    from .duel.orchestrate import Orchestrator
    from .live import LiveReporter
    from .store.writer import Store

    local = _local_dirs(args.local)
    key = Path(args.key)
    if not key.is_file():
        raise ValueError(f"no signing key at {key}; `store init` writes one, or pass --key")
    signer = Signer.from_file(key)
    store = Store(args.store, spec, signer)
    manifest = store.manifest()
    if manifest is None:
        raise ValueError(f"{args.store} is not a store; run `icil-orchestrator store init` first")
    if manifest.get("validator_key") != signer.verify_key_hex:
        raise ValueError(f"{args.store} is signed by {manifest.get('validator_key')}, not by {key}")
    if args.runtime == "local":
        from .duel.local_runtime import SubprocessPolicyRuntime

        if not local:
            raise ValueError("--runtime local serves only what --local REPO=DIR maps")
        runtime = SubprocessPolicyRuntime(spec, local)
    elif args.runtime == "weights":
        from .duel.weights_runtime import WeightsPolicyRuntime

        if spec.submission_kind != "weights":
            raise ValueError("--runtime weights serves a spec whose submission.kind is weights")
        runtime = WeightsPolicyRuntime(
            spec,
            python=args.policy_python,
            cache_root=args.cache,
            kwargs=_policy_kwargs(args.policy_kwarg),
        )
    else:
        from .duel.docker_runtime import DockerPolicyRuntime

        runtime = DockerPolicyRuntime(
            spec, cache_dir=args.cache, local=local, base_digest=args.base_image, gpus=args.gpus
        )
    token = os.environ.get(args.live_token_env) if args.live_token_env else None
    if args.live_url and not token:
        raise ValueError("--live-url needs --live-token-env naming a variable that holds the token")
    mirror = None
    if args.mirror:
        from .store.mirror import Mirror

        mirror = Mirror(store.root, args.mirror, token=os.environ.get("HF_TOKEN"))
    return Orchestrator(
        spec,
        store,
        runtime,
        args.run_dir,
        live=LiveReporter(spec, args.live_url, token),
        mirror=mirror,
    )


def _policy_kwargs(pairs: list[str] | None) -> dict[str, str]:
    """`KEY=VALUE` pairs for the validator's policy class (a weights track's `--policy-kwarg`)."""
    out: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key or key == "weights":
            raise ValueError(f"--policy-kwarg {pair!r} is not KEY=VALUE (and never weights=)")
        out[key] = value
    return out


def _logging() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


class Terminated(KeyboardInterrupt):
    """SIGTERM or SIGINT, raised where the duel or the daemon was when it arrived."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(signal.Signals(self.signum).name)


@contextmanager
def _terminable() -> Iterator[None]:
    """SIGTERM and SIGINT raise `Terminated` in the main thread, so what was running unwinds
    through its `finally` blocks - the unit's policy server or container, the benchmark's process
    group, the socket directory - before the process exits. Python's default for SIGTERM is to
    die on the spot, which leaves all of those running. A second signal while that unwinding
    happens is ignored; SIGKILL still ends the process, and the next start reaps what it left."""

    def stop(signum: int, frame: Any) -> None:
        for name in (signal.SIGTERM, signal.SIGINT):
            signal.signal(name, signal.SIG_IGN)
        raise Terminated(signum)

    previous = {name: signal.signal(name, stop) for name in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def _until_terminated(
    command: Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> int:
    try:
        with _terminable():
            return command(args)
    except Terminated as exc:
        print(
            f"stopped by {exc}: the unit that was running was torn down; the same command resumes "
            "the duel where it stopped",
            file=sys.stderr,
        )
        return 128 + exc.signum


def cmd_duel(args: argparse.Namespace) -> int:
    return _until_terminated(_duel, args)


def cmd_daemon(args: argparse.Namespace) -> int:
    return _until_terminated(_daemon, args)


def _duel(args: argparse.Namespace) -> int:
    from .duel.orchestrate import DuelFailed, DuelRequest
    from .duel.runtime import RuntimeUnavailable, SubmissionRefused
    from .ids import SubmissionRef
    from .queue import Queues
    from .store.writer import store_lock

    spec = _spec(args)
    try:
        track = args.track or spec.sole_track
        spec.track(track)
        if args.size is not None and args.size not in spec.sizes(track):
            raise ValueError(
                f"duel size {args.size!r} is not one of {', '.join(spec.sizes(track))}"
            )
        repo, _, revision = args.challenger.partition("@")
        if not repo or not revision:
            raise ValueError(f"{args.challenger!r} is not owner/name@revision")
        orchestrator = _orchestrator(args, spec)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    _logging()
    try:
        challenger = orchestrator.runtime.resolve(repo, revision)
    except (RuntimeUnavailable, SubmissionRefused, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if challenger.revision != revision:
        print(f"resolved {repo}@{revision} to {challenger.revision}", file=sys.stderr)
    store = orchestrator.store
    try:
        # A daemon holds this lock for its whole life, so a duel here never runs beside one.
        with store_lock(store.root):
            orchestrator.reap_orphans()
            head = store.head(track) or {}
            king = SubmissionRef.from_dict(head.get("king"))
            if king is not None and king.key == challenger.key:
                print(f"error: {challenger.entry} already holds the crown", file=sys.stderr)
                return 2
            baseline = spec.baseline(track)
            size = args.size
            if king is None and baseline is not None:
                # The declared baseline takes the empty throne by genesis before any entry; an
                # entrant crowned here instead would be shown as the organizer's baseline.
                if challenger.repo != baseline.get("repo"):
                    print(
                        f"error: track {track} declares a baseline, which takes its empty throne; "
                        f"run the daemon, or this duel with --challenger {baseline.get('repo')}@..., "
                        "to crown it first",
                        file=sys.stderr,
                    )
                    return 2
                size = args.size or baseline.get("size")
            queue = Queues(args.queue, spec.tracks)[track]
            head_block = int(head.get("block") or 0)
            # Numbered from the queue's counter, as the daemon numbers its duels, so the two never
            # share a run directory. The last block handed out is reused only for this very duel
            # left unfinished there: running a failed duel again resumes it.
            seed = {
                "seed_block": args.seed_block,
                "seed_block_hash": args.seed_block_hash,
            }
            req = DuelRequest(
                track, challenger, king, size, block=max(queue.block, head_block), **seed
            )
            if req.block <= head_block or not orchestrator.holds_request(req):
                req = DuelRequest(
                    track, challenger, king, size, block=queue.claim_block(head_block), **seed
                )
            result = orchestrator.run(req)
    except (DuelFailed, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    record = result.record or {}
    print(
        json.dumps(
            {
                "status": result.status,
                "kind": result.kind,
                "reason": result.reason,
                "event_id": result.event_id,
                "duel_id": result.duel_id,
                "dethroned": record.get("dethroned"),
                "king": record.get("king"),
                "new_king": record.get("new_king"),
                "king_scores": record.get("king_scores"),
                "challenger_scores": record.get("challenger_scores"),
                "void": sum(1 for u in result.units if u.get("void")),
                "units": len(result.units),
                "run_dir": str(result.run_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.published else 1


def _daemon(args: argparse.Namespace) -> int:
    from .daemon import Daemon
    from .queue import Queues

    spec = _spec(args)
    try:
        orchestrator = _orchestrator(args, spec)
        queues = Queues(args.queue, spec.tracks)
    except ValueError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    daemon = Daemon(
        orchestrator, queues, idle_sleep_s=args.idle_sleep, max_backoff_s=args.max_backoff
    )
    intake = None
    if args.admin:
        from .admin import AdminServer, read_token

        try:
            # Bound now, so a port in use stops the daemon before it takes the store; serving
            # starts once the store is held. It queues on the daemon's own queues, and publishes
            # their snapshots through the daemon, which holds the store.
            intake = AdminServer(
                spec,
                queues,
                read_token(args.admin_token_env),
                host=args.admin_host,
                port=args.admin_port,
                publish=lambda track: daemon.publish_queue(track, mirror=False),
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(
                f"error: cannot serve the intake on {args.admin_host}:{args.admin_port}: {exc}",
                file=sys.stderr,
            )
            return 2
    _logging()
    try:
        daemon.run(once=args.once, serving=None if intake is None else lambda: _serve(intake, args))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        # On a signal too: the intake stops with the loop, before the process exits.
        if intake is not None:
            intake.shutdown()
    return 0


def _serve(intake: Any, args: argparse.Namespace) -> None:
    from .admin import announce

    intake.start()
    announce(intake, args.admin_token_env)


def _add_duel_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--queue",
        default="queue",
        help="queue directory, one file per track; its block counter numbers every duel",
    )
    p.add_argument("--store", required=True, help="the store to publish to (`store init`)")
    p.add_argument(
        "--run-dir",
        required=True,
        metavar="DIR",
        help="where duels keep prompts, side results and logs, and resume from",
    )
    p.add_argument(
        "--key", default=DEFAULT_KEY, help=f"the store's signing key (default: {DEFAULT_KEY})"
    )
    p.add_argument(
        "--runtime",
        choices=("docker", "local", "weights"),
        default="docker",
        help="where policies run: docker, the sandbox (default); local, which runs submission "
        "code on this host WITHOUT A SANDBOX, as a subprocess with every permission the "
        "orchestrator has - for development with code you trust, never for a competitor's; or "
        "weights, for a spec whose submissions are weights only: the validator's own policy "
        "class serves each submission's checked weights file, and no submission code exists",
    )
    p.add_argument(
        "--policy-python",
        default=os.environ.get("ICIL_POLICY_PYTHON") or sys.executable,
        help="the policy environment's interpreter (weights runtime; default "
        "$ICIL_POLICY_PYTHON, else this one)",
    )
    p.add_argument(
        "--policy-kwarg",
        action="append",
        metavar="KEY=VALUE",
        help="a keyword for the validator's policy class, e.g. device=cuda:0 (weights runtime)",
    )
    p.add_argument(
        "--local",
        action="append",
        metavar="REPO=DIR",
        help="serve repository REPO from this directory instead of the Hub (repeatable)",
    )
    p.add_argument(
        "--cache", default="cache", help="where checkouts are kept, by commit (default: cache)"
    )
    p.add_argument(
        "--base-image",
        default=None,
        metavar="DIGEST",
        help="the base image digest while spec.json's is null (docker runtime)",
    )
    p.add_argument(
        "--gpus", type=int, default=None, help="GPUs per policy container (default: the spec's)"
    )
    p.add_argument("--live-url", default=None, help="the dashboard's base url for live frames")
    p.add_argument(
        "--live-token-env",
        default=None,
        metavar="NAME",
        help="the environment variable holding the live ingest token",
    )
    p.add_argument(
        "--mirror",
        default=None,
        metavar="REPO",
        help="also push what is published to this Hugging Face dataset (token from HF_TOKEN)",
    )


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

    ad = sub.add_parser("admin", help="the submission intake the dashboard's submit form posts to")
    ad_sub = ad.add_subparsers(dest="admin_cmd", required=True)
    ad_serve = ad_sub.add_parser(
        "serve",
        help="serve GET /admin/health and POST /admin/submissions over plain HTTP, for organizers "
        "on a private network",
    )
    ad_serve.add_argument(
        "--store",
        required=True,
        help="the store an accepted entry's queue snapshot is published to (`store init`)",
    )
    ad_serve.add_argument("--queue", default="queue", help="queue directory, one file per track")
    ad_serve.add_argument(
        "--host", default="127.0.0.1", help="address to bind (default: 127.0.0.1, loopback only)"
    )
    ad_serve.add_argument("--port", type=int, default=8799, help="port (default: 8799)")
    ad_serve.add_argument(
        "--token-env",
        default="ICIL_ADMIN_TOKEN",
        metavar="NAME",
        help="the environment variable holding the bearer token, never taken from the command "
        "line (default: ICIL_ADMIN_TOKEN)",
    )
    ad.set_defaults(func=cmd_admin)

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

    du = sub.add_parser("duel", help="run one duel (or a genesis on an empty track) and publish it")
    du.add_argument("--track", default=None, help="which track (default: the only one)")
    du.add_argument(
        "--challenger", required=True, help="owner/name@revision (a branch or tag is resolved)"
    )
    du.add_argument("--size", default=None, help="the duel size (default: the track's)")
    du.add_argument(
        "--seed-block",
        type=int,
        default=None,
        help="the chain block whose hash seeds the units (with --seed-block-hash)",
    )
    du.add_argument(
        "--seed-block-hash",
        default=None,
        metavar="0xHASH",
        help="that block's hash as the chain reports it",
    )
    _add_duel_args(du)
    du.set_defaults(func=cmd_duel)

    dm = sub.add_parser(
        "daemon", help="serve the queues: resume, genesis, duel, publish, mirror; one per store"
    )
    dm.add_argument("--once", action="store_true", help="one step for every track, then exit")
    dm.add_argument(
        "--idle-sleep", type=float, default=15.0, help="seconds to wait when every queue is empty"
    )
    dm.add_argument(
        "--max-backoff",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help="after a step crashes, wait 1s, doubling with each crash in a row up to this "
        "(default: 300)",
    )
    dm.add_argument(
        "--admin",
        action="store_true",
        help="also serve the submission intake (`admin serve`) beside the duel loop, on the "
        "daemon's own queues, from once it holds the store until it stops",
    )
    dm.add_argument(
        "--admin-host",
        default="127.0.0.1",
        help="address the intake binds (default: 127.0.0.1, loopback only)",
    )
    dm.add_argument(
        "--admin-port", type=int, default=8799, help="the intake's port (default: 8799)"
    )
    dm.add_argument(
        "--admin-token-env",
        default="ICIL_ADMIN_TOKEN",
        metavar="NAME",
        help="the environment variable holding the intake's bearer token, never taken from the "
        "command line (default: ICIL_ADMIN_TOKEN)",
    )
    _add_duel_args(dm)
    dm.set_defaults(func=cmd_daemon)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
