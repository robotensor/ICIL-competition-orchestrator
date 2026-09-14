# ICIL competition orchestrator

The orchestration layer of the RoboTensor one-demonstration in-context imitation learning (ICIL)
competition. It queues submissions, runs duels between a challenger and the reigning king, scores
them, publishes a signed result store and streams live progress to the dashboard.

It contains no benchmark. Benchmarks are separate repositories plugged in through the
`icil.benchmarks` entry point group; the first is
[ICIL-robotwin-benchmark](https://github.com/robotensor/ICIL-robotwin-benchmark). A submission is a
HuggingFace repository with runnable policy code and weights, run in a sandboxed container with no
network.

The policy protocol lives in [`packages/icil-policy`](packages/icil-policy), a distribution of its
own: a competitor's policy is served in its own process and a benchmark drives it over named
arrays.

**Status:** in progress. The first milestone plugs RoboTwin and launches a 1-arm Franka competition
with one sensorimotor demonstration per episode. The contract, benchmark discovery, the signed
store, the queue and its HTTP intake, live frames, the policy protocol, the policy sandbox and duels
are in place; the first smoke duel on RoboTwin's `franka_1arm` suite is next.

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]" -e packages/icil-policy

icil-orchestrator benchmarks list              # declared and installed benchmarks, no import
icil-orchestrator benchmarks check robotwin    # pin, ABI, catalogue, derivation, command builders

icil-orchestrator store init store/            # signing key in keys/ (generated if absent)
icil-orchestrator store verify store/ --validator-key <hex>   # signatures, events, media, schema
icil-orchestrator store mirror store/ --repo owner/dataset

icil-orchestrator queue --store store/ add owner/policy main --duel-size smoke   # resolved to its commit
icil-orchestrator queue list

python -c 'import secrets; print(secrets.token_urlsafe(32))'   # an admin token, made once
read -rs ICIL_ADMIN_TOKEN && export ICIL_ADMIN_TOKEN           # pasted, so not in shell history
icil-orchestrator admin serve --store store/   # the dashboard's submit form posts here:
                                               # 127.0.0.1:8799, bearer token

icil-orchestrator submission build-base                  # docker/policy-base, prints its digest
icil-orchestrator submission check owner/policy@main --base-image sha256:<hex>   # resolve, fetch,
                                                         # manifest, build, run, hello; reported
icil-orchestrator submission check local/replay@main --local packages/icil-policy/examples/replay_policy
icil-orchestrator submission prune                       # the icil-submission images nothing uses

icil-orchestrator duel --track franka_1arm --challenger owner/policy@main --size smoke \
    --store store/ --run-dir runs/ --base-image sha256:<hex>   # one duel, or genesis, published
icil-orchestrator daemon --store store/ --run-dir runs/ --queue queue/ \
    --live-url https://dashboard --live-token-env ICIL_LIVE_TOKEN   # serve the queues
icil-orchestrator daemon --store store/ --run-dir runs/ --queue queue/ --admin   # and the intake,
                                               # as admin serve does, in the same process
```

A submission is a Hugging Face repository at a commit: `queue add` resolves a branch or a tag to
its sha once, through the Hub, confirms a sha it is given, and everything published hangs off that
sha. `submission check` fetches it into `cache/<sha>/` (git-ignored), reads its `icil.yaml` as a
plain file, builds its image `FROM` the pinned base by digest with its checkout copied in and its
requirements installed at build time (bounded by `--build-timeout`), runs it under
`spec.submission.sandbox` - no network, a read-only root, `/tmp` a nosuid,nodev tmpfs of
`tmpfs_bytes` that may run code, a non-root user, memory with no swap, cpu and pid limits, one
directory shared for the socket (a small noexec tmpfs when the orchestrator is root) - and says
`hello`.
A step that fails is the submission's rejection with the reason or the harness's error, and the
container is removed either way; a container whose process was killed is removed by the next
start. `--local DIR` takes a directory in the Hub's place. See
[`docker/policy-base`](docker/policy-base/README.md) for the base image, what a policy finds at
run time and what was measured.

A policy may compile at run time - `torch.compile`, Triton, cffi, `torch.utils.cpp_extension`
(with `ninja` in its requirements):
the base is CUDA's `devel` image with gcc, g++, make and Python's headers, and `HOME`, `TMPDIR`,
`XDG_CACHE_HOME` and the Triton, inductor and torch extension caches all point into `/tmp`, where
what they build may be loaded. Submission code already runs natively in its container, so letting
it run code it wrote to a size-capped nosuid,nodev tmpfs removes a speed bump rather than a
boundary; the boundary is the network (none), the read-only root, the non-root user and the
limits.

A duel fetches and checks both sides, then has the benchmark materialize every unit's prompt
once, before either side runs: both run from those files, and the event publishes each prompt's
sha256. Each unit gets a freshly served policy - a container per unit under `--runtime docker`,
the default - and the benchmark's unit command drives it over the policy socket. Every unit's
result is written to the run directory as it finishes, so a duel or a daemon that is killed and
started again resumes where it stopped, running no unit twice. The first entrant of an empty track
is crowned by genesis (or the track's declared baseline, by the daemon). `--runtime local --local
REPO=DIR` serves a directory's policy as a subprocess on the host, with no sandbox at all: for
development with code you trust, never for a competitor's.

The unit command is given the unit's time limits: `act_timeout_s`; `unit_timeout_s`, the seconds
its subprocess has before it is killed, so it can write why a unit it cannot finish ended; and
`policy_budget_s`, what starting the policy left of `budgets.policy_budget_seconds` for all of its
calls together, never more than that timeout less the benchmark's own result reserve. It is also
given `policy_log`, a copy of the end of the policy's log kept beside its result while it runs,
never the log file the policy itself writes. It runs with an allow-listed environment: the locale,
the interpreter's paths, what a GPU simulator reads, RoboTwin's `ROBOTWIN_ICIL_PYTHON` and
`ROBOTWIN_ICIL_DENOISER`, and never `HF_TOKEN` or the live token. A benchmark may choose a unit's
scene only while it materializes (RoboTwin's expert tries the unit's candidate seeds in order), so
a published unit's `instance_params.scene_seed` is the seed its prompt was built on.

What a unit counts as depends on whose doing its end was:

- **Void, for both sides**, only for a harness cause: its prompt failed to materialize or was
  rejected, or is not the file both sides ran from (a benchmark reporting another
  `prompt_sha256`, or another scene seed than the prompt's own), the benchmark crashed or ran out
  of the unit's time while the policy was within its budget (or reported `void_cause: "harness"`),
  Docker or the host failed (a container removed from outside, a `docker inspect` that does not
  answer in time), or the orchestrator stopped the run. A duel with more than `max_void_fraction`
  of its units void is void and publishes nothing; once that is certain, nothing more is
  materialized or played.
- **That side's failure** for anything its own submission did: its policy container exiting on its
  own or OOM-killed in its sandbox (read from `docker inspect`), not listening within
  `budgets.policy_start_seconds`, taking all of `budgets.policy_budget_seconds` to start or using
  it up over the unit's calls, an act timeout, an error reply or a non-zero exit (a benchmark
  reports those as `void_cause: "policy"`). A scored result is never turned into a void
  afterwards, and a policy that died on one unit is simply served again for the next.
- **A refused king forfeits**: when the king's repository is gone or private, its image no longer
  builds or its manifest is invalid, every king unit is a failure and the duel is published with
  the note `king forfeit: <reason>`; the challenger takes the crown if its own average clears the
  margin. A refused challenger is refused: nothing is published and its entry is used up.

A duel is never published against a king who no longer holds the crown: before it starts or
resumes, and again before it publishes, the request's king is compared with the track head's, and a
stale duel's run is moved aside as `<dir>.stale-<n>` while the daemon puts its challenger back at
the head of the queue. A duel resumed after its record was appended rebuilds `head.json` from the
index and pushes its files to the mirror again. `duel` numbers its block from the queue's counter
(`--queue`) and refuses to run while a daemon holds the store's lock. The daemon keeps a duel in
progress while the harness is only unavailable (benchmark not installed, Docker down, the Hub
unreachable), moves aside a run directory holding another request, and backs off exponentially,
up to `--max-backoff` (300 s), when a step keeps crashing. SIGTERM and SIGINT tear the running
unit's policy and benchmark down before exiting; after a SIGKILL, the next start reaps what was
left - the policy containers whose owner process is gone (the sandbox labels each container with
the process that started it), and the process groups recorded in each unit's `pids.json` - before
any unit runs again. A unit's container runs with exactly the sandbox's `docker run`: the duel
adds nothing to it.

Deploy the dashboard before this orchestrator: its live ingest must accept the `materializing`
phase, which every duel reports between `checking` and `evaluating`, or those frames are refused.

`admin serve` is the intake the dashboard's dev-mode submit form posts to (the dashboard's
`ICIL_ADMIN_URL` and `ICIL_ADMIN_TOKEN`): `GET /admin/health` answers
`{ok, spec_version, tracks, queue_lengths}`, and `POST /admin/submissions` takes
`{repo, revision, track, duel_size, source}`, resolves the revision as `queue add` does (null is the
repository's default branch; a ref under `refs/`, a pull request's included, is refused), queues the
entry in `--queue` and publishes the track's queue snapshot to `--store`. It answers with the
entry's `key`, commit `revision`, `entry`, `position` and `accepted_at`; a submission already
waiting answers with the place it has and queues nothing, or 409 if it waits at another duel size.
It is for organizers on a private network, not for competitors and not for the internet: plain
HTTP, bound to `127.0.0.1` unless `--host` says otherwise, and every request carries exactly one
`Authorization: Bearer <token>`. The token is read from the environment variable `--token-env`
names (`ICIL_ADMIN_TOKEN` by default) and never from the command line; it must be 32 or more
printable ASCII characters with no space, it is compared in constant time and never logged, and
without it the server does not start. A request has 10 s to arrive whole, with headers of at most
16 KB and a JSON body of at most 8 KB with a `Content-Length` (chunked is refused); at most 64
connections are served at once. An unknown field, track or duel size is refused before the Hub is
asked. The Hub refusing a revision is a 422 with its reason, a Hub that cannot be asked a 503. The
Hub is given 5 s for each step of its request and resolving 5 s in all: past that nothing is queued
and the answer is 503, because the dashboard, waiting `ICIL_ADMIN_TIMEOUT_MS` (6 s by default), has
already reported a timeout.

`daemon --admin` serves the same intake beside the duel loop, in the daemon's process and on its
queues, with `admin serve`'s defaults (`--admin-host`, `--admin-port`, `--admin-token-env`). It
binds before the daemon takes the store, so a port in use stops the daemon at once, and it serves
only once the store is held. An accepted entry's queue snapshot is written to the store at once,
even while a duel runs, and goes to the mirror with the daemon's next push. It stops with the
daemon: after `--once`, when the store cannot be taken, and on SIGTERM or SIGINT. `admin serve`
beside a running daemon queues on the same files, but cannot publish into the store the daemon
holds, so its snapshot waits until the daemon next takes or settles an entry.

`store init` writes the store's ed25519 signing key to `keys/orchestrator.ed25519` (mode 0600)
unless `--key` says otherwise. It is the only thing that can publish as this store, so keep it out
of the checkout and off the mirror: `/keys/` and `/queue/` are git-ignored, and the mirror uploads
the store's layout and nothing beside it.

`tests/fixtures/store` is a small signed history for rendering the dashboard
(`ICIL_STORE=$PWD/tests/fixtures/store npm run dev` in the dashboard); it is not a result.
