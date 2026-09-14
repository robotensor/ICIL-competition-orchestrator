"""The submission intake: the HTTP endpoint the dashboard's dev-mode submit form queues through.

Ported from icilval's `admin.py` (`robotensor/ICIL-LiberoGen-bench`, `milestone-two-fields`) for
submissions that are code. A revision is resolved to its commit through the Hub, once, by
`submissions.resolve` - the queue-time resolution `queue add` does - and the entry goes on its
track's queue beside the ones `queue add` puts there. There is no model config check, so there is
no flag to skip one.

    GET  /admin/health       200 {ok, spec_version, tracks, queue_lengths}
    POST /admin/submissions  {repo, revision: str|null, track?, duel_size?: str|null, source?}
                             200 {ok, queued, track, key, repo, revision, entry, duel_size,
                                  position, accepted_at, source, message}
    a refusal                4xx {ok: false, error, field?}; 503 when the Hub cannot be asked

Every request carries `Authorization: Bearer <token>`. This is a tool for organizers on a private
network: plain HTTP, bound to loopback unless told otherwise, holding a credential that queues GPU
time. So it does not start without a token, compares it in constant time and never logs it, and it
reads a body only when the request says how long it is and that is at most `MAX_BODY_BYTES`.

The dashboard is the other side of this contract (`lib/dev.ts`, `app/api/dev/submit/route.ts`). It
reads `error` from a refusal and nothing else of it; `key`, `revision`, `entry`, `position`,
`accepted_at` and `message` from an acceptance; and it stops waiting after `ICIL_ADMIN_TIMEOUT_MS`,
6 s unless set, which is why the Hub is given `RESOLVE_TIMEOUT_S`.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import sys
import threading
import traceback
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .ids import SubmissionRef, is_repo
from .queue import Queues
from .spec import Spec
from .store.writer import Store, store_lock
from .submissions.errors import SubmissionError, SubmissionRejected
from .submissions.resolve import resolve

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8799
DEFAULT_TOKEN_ENV = "ICIL_ADMIN_TOKEN"

HEALTH_PATH = "/admin/health"
SUBMIT_PATH = "/admin/submissions"

#: A submission is a few hundred bytes of JSON. The dashboard's own route caps its body at the same.
MAX_BODY_BYTES = 8 * 1024

#: A refused body up to this long is read and dropped, never kept, so that a client still sending
#: it reads the refusal instead of a connection reset. A longer one is left to the closing socket.
DISCARD_BYTES = 64 * 1024

#: What a submission may say, and nothing else. An unknown field is refused rather than ignored, so
#: a client that believes it set something (a check to skip, a second track) hears that it did not.
FIELDS = ("repo", "revision", "track", "duel_size", "source")

#: The source an entry is queued with when the request names none (`queue add` queues with "cli").
DEFAULT_SOURCE = "admin"

#: Seconds the Hub is given for each step of resolving a revision (connect, send, read). Under the
#: dashboard's default wait, so a Hub that hangs is answered as unavailable while the form still
#: listens, rather than timing the form out on an entry that may yet be queued after it gave up.
RESOLVE_TIMEOUT_S = 5.0

#: Seconds a connection may take to send its request, and may sit idle between two.
REQUEST_TIMEOUT_S = 10.0

#: A branch, a tag or a commit sha, as a Hub revision: printable ASCII with no space.
REVISION_RE = re.compile(r"[\x21-\x7e]{1,255}")
SOURCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
CONTENT_LENGTH_RE = re.compile(r"[0-9]{1,16}")

#: A log line is scrubbed of the token and cut to this length.
MAX_LOG_CHARS = 2000


class Refused(Exception):
    """A request answered `status` with `error`, which names what to fix; nothing was queued."""

    def __init__(
        self,
        status: int,
        error: str,
        field: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(error)
        self.status = status
        self.error = error
        self.field = field
        self.headers = dict(headers or {})

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"ok": False, "error": self.error}
        if self.field:
            body["field"] = self.field
        return body


class HubApi:
    """`HfApi.repo_info` with a timeout, which `resolve` does not pass and the Hub client does not
    set: without one a Hub that stops answering holds the request until the dashboard gives up."""

    def __init__(self, timeout_s: float = RESOLVE_TIMEOUT_S) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi()
        self.timeout_s = timeout_s

    def repo_info(self, repo_id: str, **kwargs: Any) -> Any:
        return self._api.repo_info(repo_id, timeout=self.timeout_s, **kwargs)


class AdminServer:
    """The intake on its own threads: `serve_forever` in the foreground, or `start` beside a loop.

    `api` is what `resolve` asks (an `HfApi`, or a stand-in); `store`, when given, is where an
    accepted entry's queue snapshot is published, as `queue --store add` publishes it.
    """

    def __init__(
        self,
        spec: Spec,
        queues: Queues,
        token: str,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        store: Store | None = None,
        api: Any = None,
        request_timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        if not token or not token.strip():
            raise ValueError("the admin token is empty; the intake does not serve without one")
        self.spec = spec
        self.queues = queues
        self.store = store
        self.api = api if api is not None else HubApi()
        self._token = token
        self._token_bytes = token.encode("utf-8", "surrogateescape")
        #: Queueing and publishing the snapshot, one request at a time within this process; the
        #: queue file's own lock stands between this process and any other.
        self._lock = threading.Lock()
        server_class = _IPv6Server if ":" in host else ThreadingHTTPServer
        self.httpd = server_class((host, port), _handler(self, request_timeout_s))
        self.httpd.daemon_threads = True
        self.host = str(self.httpd.server_address[0])
        self.port = int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    # -- what a request is checked against --------------------------------------------------

    def authorized(self, header: str | None) -> bool:
        scheme, _, presented = (header or "").strip().partition(" ")
        presented = presented.strip()
        if scheme.lower() != "bearer" or not presented:
            return False
        return hmac.compare_digest(presented.encode("utf-8", "surrogateescape"), self._token_bytes)

    def scrub(self, text: str) -> str:
        """`text` without the token in it. Nothing logged carries a header, a body or a query, so
        this is the second guard, for a token a client sent somewhere it does not belong."""
        return text.replace(self._token, "[redacted]")

    # -- the two answers --------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "spec_version": self.spec.version,
            "tracks": list(self.spec.tracks),
            "queue_lengths": {track: len(queue.entries()) for track, queue in self.queues.items()},
        }

    def submit(self, body: Any) -> dict[str, Any]:
        """Queue what `body` names and say where it is, or raise `Refused` with nothing queued."""
        repo, revision, track, duel_size, source = self.validate(body)
        sha = self.resolve(repo, revision)
        with self._lock:
            entry, position, queued = self.queues[track].offer(
                repo, sha, duel_size=duel_size, source=source
            )
            if queued:
                self.publish(track)
        log.info(
            "%s submission key=%s repo=%s sha=%s source=%s track=%s position=%d",
            "accepted" if queued else "already queued",
            entry.key,
            entry.repo,
            entry.revision,
            source,
            track,
            position,
        )
        if queued:
            message = (
                f"Queued at position {position} for {track}. The orchestrator duels it against the "
                "reigning policy when its turn comes."
            )
        else:
            message = (
                f"Already queued at position {position} for {track}, since {entry.accepted_at}; "
                "nothing new was queued."
            )
        return {
            "ok": True,
            "queued": queued,
            "track": track,
            "key": entry.key,
            "repo": entry.repo,
            "revision": entry.revision,
            "entry": entry.ref.entry,
            "duel_size": entry.duel_size,
            "position": position,
            "accepted_at": entry.accepted_at,
            "source": entry.source,
            "message": message,
        }

    def validate(self, body: Any) -> tuple[str, str | None, str, str | None, str]:
        """`(repo, revision, track, duel_size, source)` from a request body, checked against the
        spec before anything is asked of the Hub."""
        if not isinstance(body, dict):
            raise Refused(400, "The body must be a JSON object.", "body")
        unknown = sorted(set(body) - set(FIELDS))
        if unknown:
            names = ", ".join(repr(name[:40]) for name in unknown[:5])
            raise Refused(
                400,
                f"Unknown field {names}; a submission takes {', '.join(FIELDS)} and nothing else.",
                "body",
            )

        repo = body.get("repo")
        if not isinstance(repo, str) or not is_repo(repo):
            raise Refused(422, "repo must be a Hugging Face repo id, owner/name.", "repo")

        revision = body.get("revision")
        if revision is not None and (
            not isinstance(revision, str) or not REVISION_RE.fullmatch(revision)
        ):
            raise Refused(
                422,
                "revision must be a commit sha, a branch or a tag, or null for the repository's "
                "default branch.",
                "revision",
            )

        track = body.get("track")
        if track is None:
            try:
                track = self.spec.sole_track
            except ValueError as exc:
                raise Refused(422, f"track is required: {exc}.", "track") from None
        elif not isinstance(track, str) or track not in self.spec.tracks:
            raise Refused(
                422, f"Unknown track; the tracks are {', '.join(self.spec.tracks)}.", "track"
            )

        duel_size = body.get("duel_size")
        sizes = self.spec.sizes(track)
        if duel_size is not None and (not isinstance(duel_size, str) or duel_size not in sizes):
            raise Refused(
                422,
                f"duel_size must be one of {', '.join(sizes)} for {track}, or null for its default.",
                "duel_size",
            )

        source = body.get("source")
        if source is None:
            source = DEFAULT_SOURCE
        elif not isinstance(source, str) or not SOURCE_RE.fullmatch(source):
            raise Refused(
                422,
                "source must be a name of at most 64 letters, digits and '.', '_', ':' or '-'.",
                "source",
            )
        return repo, revision, track, duel_size, source

    def resolve(self, repo: str, revision: str | None) -> str:
        """The commit `repo@revision` names on the Hub, through `submissions.resolve`. No revision
        is the default branch, `main` on the Hub (huggingface_hub's `DEFAULT_REVISION`)."""
        if revision is None:
            from huggingface_hub.constants import DEFAULT_REVISION

            revision = DEFAULT_REVISION
        try:
            return resolve(repo, revision, api=self.api).sha
        except SubmissionRejected as exc:
            raise Refused(422, exc.reason) from None
        except SubmissionError as exc:
            raise Refused(503, str(exc)) from None

    def publish(self, track: str) -> None:
        """`tracks/{track}/queue.json` in the store, as `queue --store add` writes it. A store held by
        a running orchestrator is left to it: it rewrites the snapshot every cycle."""
        if self.store is None:
            return
        try:
            with store_lock(self.store.root):
                head = self.store.head(track) or {}
                king = SubmissionRef.from_dict(head.get("king"))
                schema = int(self.spec.store["schema"])
                self.store.write_queue(track, self.queues[track].snapshot(track, king, schema))
        except RuntimeError as exc:
            log.info("queue snapshot left to the orchestrator: %s", exc)
        except OSError as exc:
            log.warning("queue snapshot for %s not published: %s", track, exc)

    # -- lifecycle --------------------------------------------------------------------------

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        finally:
            self.httpd.server_close()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.httpd.serve_forever, name="icil-admin", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class _IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object naming a field twice is refused: which of the two counts is a guess."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise Refused(400, f"The body names {key[:40]!r} twice.", "body")
        out[key] = value
    return out


def _no_constant(name: str) -> Any:
    raise Refused(400, f"{name} is not JSON.", "body")


def _handler(server: AdminServer, timeout_s: float) -> type[BaseHTTPRequestHandler]:
    routes = {HEALTH_PATH: "GET", SUBMIT_PATH: "POST"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "icil-orchestrator-admin"
        sys_version = ""
        timeout = timeout_s

        def do_GET(self) -> None:
            self._handle()

        do_POST = do_PUT = do_PATCH = do_DELETE = do_GET

        # -- one request ----------------------------------------------------------------------

        def _handle(self) -> None:
            self._body_read = False
            close = False
            headers: dict[str, str] = {}
            try:
                status, body, close = self._answer()
            except Refused as exc:
                status, body, headers, close = exc.status, exc.body(), exc.headers, True
                self._discard_body()
            except Exception:  # noqa: BLE001 - a bug answers 500 rather than dropping the socket
                log.error("admin request failed:\n%s", server.scrub(traceback.format_exc()))
                status, close = 500, True
                body = {"ok": False, "error": "The intake failed on this request; see its log."}
                self._discard_body()
            try:
                self._reply(status, body, headers, close)
            except OSError:
                self.close_connection = True

        def _answer(self) -> tuple[int, dict[str, Any], bool]:
            if not server.authorized(self.headers.get("Authorization")):
                raise Refused(
                    401,
                    "The bearer token is missing or wrong.",
                    headers={"WWW-Authenticate": 'Bearer realm="icil-orchestrator"'},
                )
            path = self.path.split("?", 1)[0]
            method = routes.get(path)
            if method is None:
                raise Refused(404, "Not found.")
            if self.command != method:
                raise Refused(405, f"{path} takes {method}.", headers={"Allow": method})
            if path == HEALTH_PATH:
                # A body on a GET is not read, so the connection it came on is not reused.
                sent_body = "Transfer-Encoding" in self.headers or self.headers.get(
                    "Content-Length", "0"
                ).strip() not in ("", "0")
                return 200, server.health(), sent_body
            return 200, server.submit(self._read_json()), False

        def _read_json(self) -> Any:
            """The body, read to its declared length and no further, as JSON."""
            if "Transfer-Encoding" in self.headers:
                raise Refused(411, "A chunked body is refused; send Content-Length.", "body")
            lengths = self.headers.get_all("Content-Length") or []
            if not lengths:
                raise Refused(411, "Content-Length is required.", "body")
            if len(lengths) != 1 or not CONTENT_LENGTH_RE.fullmatch(lengths[0].strip()):
                raise Refused(400, "Content-Length must be one decimal number.", "body")
            length = int(lengths[0].strip())
            if length > MAX_BODY_BYTES:
                raise Refused(
                    413,
                    f"The body is {length} bytes; a submission is at most {MAX_BODY_BYTES}.",
                    "body",
                )
            try:
                raw = self.rfile.read(length)
            except TimeoutError:
                raise Refused(408, "The body did not arrive in time.", "body") from None
            self._body_read = True
            if len(raw) != length:
                raise Refused(400, "The body ended before its Content-Length.", "body")
            try:
                return json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=_no_duplicates,
                    parse_constant=_no_constant,
                )
            except (UnicodeDecodeError, ValueError, RecursionError):
                raise Refused(400, "The body is not valid JSON.", "body") from None

        def _discard_body(self) -> None:
            if self._body_read or "Transfer-Encoding" in self.headers:
                return
            length = (self.headers.get("Content-Length") or "").strip()
            if not CONTENT_LENGTH_RE.fullmatch(length) or int(length) > DISCARD_BYTES:
                return
            self._body_read = True
            remaining = int(length)
            try:
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 16 * 1024))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:
                self.close_connection = True

        def _reply(
            self, status: int, body: dict[str, Any], headers: Mapping[str, str], close: bool
        ) -> None:
            data = (json.dumps(body) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in headers.items():
                self.send_header(name, value)
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        # -- logging: a method, a path and a status; never a header, a body or a query ---------

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            path = (getattr(self, "path", "") or "-").split("?", 1)[0]
            self._log("%s %s %s %s", self.client_address[0], self.command, path[:200], int(code))

        def log_error(self, format: str, *args: Any) -> None:
            self._log(format, *args)

        def log_message(self, format: str, *args: Any) -> None:
            self._log(format, *args)

        def _log(self, format: str, *args: Any) -> None:
            if log.isEnabledFor(logging.INFO):
                log.info("%s", server.scrub(format % args)[:MAX_LOG_CHARS])

    return Handler


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(
    spec: Spec,
    *,
    store_dir: str | Path,
    queue_dir: str | Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token_env: str = DEFAULT_TOKEN_ENV,
    environ: Mapping[str, str] | None = None,
) -> int:
    """`icil-orchestrator admin serve`. 2 for what stops it starting; 0 when interrupted."""
    environ = os.environ if environ is None else environ
    token = (environ.get(token_env) or "").strip()
    if not token:
        print(
            f"error: {token_env} is not set; the intake does not serve without a bearer token",
            file=sys.stderr,
        )
        return 2
    store = Store(store_dir, spec)
    if store.manifest() is None:
        print(
            f"error: {store_dir} is not a store; run `icil-orchestrator store init` first",
            file=sys.stderr,
        )
        return 2
    try:
        server = AdminServer(
            spec, Queues(queue_dir, spec.tracks), token, host=host, port=port, store=store
        )
    except (OSError, ValueError) as exc:
        print(f"error: cannot serve on {host}:{port}: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not _loopback(server.host):
        log.warning(
            "%s is not a loopback address: the token and every submission cross the network in "
            "plain HTTP; serve it only on a private network",
            server.host,
        )
    log.info("admin intake listening on %s (bearer token from $%s)", server.url, token_env)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
