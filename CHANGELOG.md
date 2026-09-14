# Changelog

## Unreleased

- (docs): the track's three skills are the benchmark's surveyed `franka_1arm` suite, no longer
  provisional; stack_bowls_two, an arm-switching task, stands in for stacking.
- (feat): `spec.json` (spec_version 7) and `store-schema.json` (schema 4) for one track,
  `franka_1arm`, on the `robotwin` benchmark's `franka_1arm` suite. Its three skills are
  provisional until the Franka expert survey. A `submission` block (manifest api, policy protocol,
  base image, sandbox limits) replaces `model` and `architecture`; `pools` is gone.
- (feat): benchmark plugin ABI v1 in `icil_orchestrator.benchmarks`: discovery through the
  `icil.benchmarks` entry point group with the spec's distribution, version and wheel sha256 pin
  checked before import; `derive_units(..., category)`; `run_command(..., authkey_env)`.
- (feat): a benchmark's unit command runs as a subprocess; a crash, timeout or missing
  `result.json` voids that unit with the reason and the rest still run.
- (feat): `icil-orchestrator benchmarks list|check <id>`.
- (feat): the signed store (writer, verify, Hugging Face mirror), the per-track challenger queue
  and live frames, in the layout and live schema 4 the dashboard already reads.
- (feat): `icil-orchestrator store init|verify|mirror` and `queue add|list|remove`.
- (feat): every index record signs the sha256 of its event file (`event_sha256`, an optional
  `IndexRecord` property of schema 4), so a published event's units, prompt hashes and clip hashes
  cannot be edited unnoticed. `store verify` checks it, re-hashes every clip against its content
  address, rebuilds `head.json` and the track list from the signed records rather than the
  unsigned manifest, names the line any bad byte broke, and takes `--validator-key` to be told
  which key must have signed; it prints the key it trusted.
- (feat): `run_units` takes the side's wall clock (`budgets.side_wall_seconds`): units not started
  by then are void as timed out.
- (fix): a benchmark subprocess gets an allow-listed environment (never `HF_TOKEN` or the live
  token) and its process group is killed however its unit ends; a stale `result.json` or
  `evaluation.mp4` in its directory is cleared before it runs.
- (fix): a benchmark module must resolve to a file of its pinned distribution, and with a wheel
  pin the installed files must still match their RECORD hashes.
- (fix): `queue add` takes a resolved 40-hex commit sha only; queue files are locked and a queue
  file that cannot be read is refused instead of being taken for an empty queue.
- (fix): `store mirror` uploads the store's layout and nothing beside it (never a signing key),
  refuses a root that is not a store, and drops the `--all` flag, which did nothing.
- (test): a fake benchmark distribution under `tests/fake_benchmark` and a reproducible fixture
  store under `tests/fixtures/store`.
- (feat): `packages/icil-policy`, the policy protocol: a competitor's policy named by `icil.yaml`
  and served by `python -m icil_policy.serve` to one client, which a benchmark drives with
  `icil_policy.client.RemotePolicy`. Named arrays and JSON fields over an authenticated socket, no
  pickling, object dtypes refused at both ends, hard per-call timeouts, and one
  `PolicyUnavailable` for every failure, `close` apart, which is best effort. Both ends bound what
  a hostile message can cost them: at most 1024 arrays and 32 dimensions each, keys of at least 16
  bytes, an `icil.yaml` of at most 1 MiB whose problems quote only excerpts, and a session that
  ends after `--idle-timeout-s` of silence. Examples `replay_policy` and `zero_policy` are
  complete competitor repositories. Depends on numpy and PyYAML.
- (feat): a submission runs pinned to its commit, with no network. `queue add` resolves a branch
  or a tag to its commit sha through the Hub, once; the checkout is fetched into a cache addressed
  by that sha, no larger than `max_repo_bytes`; `icil.yaml` and its requirements must be plain
  files of the repository (no symbolic link, no pipe) before `icil_policy` reads them;
  `docker/policy-base/Dockerfile` (CUDA 12.8 runtime, Python 3.10, icil-policy) is built by
  `submission build-base` and referenced by digest; a submission's image is its checkout and its
  requirements installed at build time, nothing else; it runs under exactly
  `spec.submission.sandbox` - `--network none`, `--read-only`, `--tmpfs /tmp`, the non-root user,
  the GPU count and the memory, cpu and pid limits - with one directory mounted for the socket, the
  authkey passed by variable name, and `hello` within `budgets.policy_start_seconds` as the health
  check; the container is removed whatever happened. `icil-orchestrator submission check
  <repo>@<revision> [--local DIR]` reports every step; a manifest naming a missing class or
  requirements that do not install is a rejection with the reason and nothing runs.
- (fix): the policy sandbox after review. The container gets no swap (`--memory-swap` equal to
  `--memory`), so `memory_bytes` is its total. The one directory it shares with the host is a
  64 MiB, 64-entry tmpfs mounted by the orchestrator when root, so a policy cannot fill the host's
  disk through it, and only a socket itself (not a link at its name) counts as listening. The
  build is bounded (`submission check --build-timeout`, 1800 s by default until `spec.budgets`
  carries a build budget). A build that fails or runs past it is a rejection once the
  orchestrator's own index probe (pip in the base, with nothing of the submission, never cached)
  gets through, whatever the build printed, and the harness's error when the probe cannot reach
  the index either. The session kept after `hello` is bounded by
  `act_timeout_s`. A file the Hub declares no size for is not downloaded; a missing branch or
  repository is rejected with the Hub's words for it; `queue add` confirms a commit sha on the Hub
  too; `--local` holds the ref to a repo id. Containers are labelled with the process that started
  them, and a start reaps those whose process has ended, with their tmpfs.
- (feat): `icil-orchestrator submission prune` removes the `icil-submission` images no container
  uses. The checkout is copied into its image rather than bind-mounted read-only as issue #4's
  scope put it: the requirements install needs it at build time, the recorded image id is the code
  that ran, and the container mounts nothing of the host but its socket directory.
- (feat): a policy may compile at run time - `torch.compile`, Triton, cffi,
  `torch.utils.cpp_extension` (with `ninja` in its requirements). The sandbox's `/tmp` is a tmpfs mounted `exec,nosuid,nodev` and
  capped, by two keys added beside `spec.submission.sandbox.tmpfs`, which keeps its shape:
  `tmpfs_exec` (true) and `tmpfs_bytes` (8 GiB for each path, all of them together no more than
  `memory_bytes`, which their pages count against); `spec_version` stays 7. `validate_spec` also
  refuses a tmpfs path Docker would not take as spelled, a repeated one, and one at `/` or at or
  under `/proc`, `/sys`, `/dev`, the checkout or the socket's directory. The container starts with `HOME=/tmp/home`, made at start,
  and `TMPDIR`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and
  `TORCH_EXTENSIONS_DIR` pointing into `/tmp`. `docker/policy-base` is built from CUDA 12.8.1
  `devel` (nvcc and the CUDA headers) with build-essential and python3.10-dev: 9.49 GB, where the
  runtime base was 3.52 GB. Submission code already runs natively in its container, so exec on a
  capped nosuid,nodev tmpfs removes a speed bump, not a boundary: no network, the read-only root,
  the non-root user and the limits are unchanged. The shared socket directory's tmpfs is now
  mounted `nosuid,nodev,noexec` as well, so the spec's tmpfs is the only place code a policy
  writes can run from. `tests/test_submission_jit.py` serves a policy that compiles C on its
  first act and, under `slow`, one whose act runs `torch.compile` on the CPU, and walks every
  mount inside to check where the sandbox user can write and run code.
- (feat): a duel runs from queue entry to published record, both sides on the same
  demonstrations (`icil_orchestrator.duel`, `icil_orchestrator.daemon`). Phases `fetching ->
  checking -> materializing -> evaluating(challenger) -> evaluating(king) -> publishing -> done |
  failed`, each a live frame; `materializing` is a new live phase (schema `LiveFrame.phase`), which
  the dashboard does not accept yet. Every unit's prompt is produced once through the plugin's
  `materialize_command` before either side runs, verified with `verify_prompt`, and published as
  `prompt_sha256` (the sha256 of the file's bytes); a unit whose prompt fails is void for both
  sides. A submission's policy is reached only through `duel.runtime.PolicyRuntime` - `resolve`,
  `fetch`, `prepare` (manifest, image, a `hello`) and `serve`, a fresh policy per unit -
  implemented over the sandbox by `duel.docker_runtime` and, with no sandbox, for development, by
  `duel.local_runtime`. Scoring: per-skill success rates over non-void units, their mean, the
  crown to the challenger iff it beats the king's mean by `score_margin` points; a unit void on
  either side is void for both, and a duel with more than `max_void_fraction` void is void and
  publishes nothing. A unit is void only for a harness cause; what a side's own submission does -
  its container exiting or OOM-killed, never listening, an act timeout or error, a benchmark's
  `void_cause: "policy"` - is that side's failure, and a policy that died is served again for the
  next unit. A refused challenger is refused and publishes nothing; a refused king forfeits every
  unit and the duel is published with the note `king forfeit: <reason>`. An empty track crowns its
  first challenger (or its declared baseline) by genesis, with its own scores. A duel resumes from
  its run directory - prompts, each side's `results.jsonl`, the index checked for its own event -
  so a killed daemon restarted runs every unit once per side and publishes one record. The event
  also carries both sides' commits and image digests, the benchmark's `info()` and pin, and the
  scoring. The dashboard must be deployed first, to accept the `materializing` phase.
- (fix): a duel is never published against a king who lost the crown while it was stopped: its run
  is moved aside as `<dir>.stale-<n>` and its challenger queued again at the head. A duel resumed
  after its record was appended rebuilds `head.json` from the index and pushes to the mirror again.
- (fix): SIGTERM and SIGINT tear the running unit's policy and benchmark down before `duel` or
  `daemon` exits (128+signal); after a SIGKILL, the next start reaps the policy containers whose
  owner process is gone and the process groups in each unit's `pids.json` before a unit runs again.
- (fix): a duel's containers run with exactly the sandbox's `docker run` - the scratch tmpfs that
  may run code, `HOME` and the JIT caches in it, the noexec socket tmpfs - and the docker runtime
  adds nothing to it. The sandbox's owner labels and `reap_orphans` replace the duel's own store
  labels and reaping, and the runtime seam's `bind` is gone. Only a socket itself counts as a
  unit's policy listening; whether its container was OOM-killed comes from `Docker.state`
  (`ContainerState.oom_killed`); and a `docker inspect` that times out voids its unit for the
  harness instead of failing the duel, while a unit already scored keeps its score.
- (fix): `duel` numbers its block from the queue's counter (`--queue`) and refuses to run beside a
  daemon; it leaves a declared baseline's empty throne to the daemon. The daemon keeps a duel in
  progress while the harness is unavailable, moves aside a run directory holding another request,
  and backs off exponentially up to `--max-backoff` when a step keeps crashing.
- (fix): each side gets at most an even share of what materializing left of the duel's wall clock,
  materializing stops at the duel's deadline, serving a policy counts in its unit's budget, and a
  prompt is re-checked after its unit and against the hash the benchmark read.
- (feat): a unit's benchmark is told its time limits and given the policy's log, under the names
  the RoboTwin plugin reads: `unit_timeout_s`, the seconds before its subprocess is killed, so it
  writes why a unit it cannot finish ended; `policy_budget_s`, what starting the policy left of
  the new `budgets.policy_budget_seconds` (300, below `unit_wall_seconds`), never more than that
  timeout less the plugin's `info()["limits"]["result_reserve_s"]`; and `policy_log`, a copy of
  the end of the policy's log in the unit's directory, never the log the policy writes. A policy
  that uses up its budget, or takes all of it to start, fails the unit rather than running it
  into a void for both sides.
- (feat): a published unit's `instance_params.scene_seed` is the seed its prompt was built on, as
  the materialize result names it, and a result naming another seed than `verify_prompt` read from
  the file voids the prompt: RoboTwin chooses the scene among a unit's candidates only then.
- (fix): a benchmark subprocess keeps `ROBOTWIN_ICIL_PYTHON` and `ROBOTWIN_ICIL_DENOISER` from the
  orchestrator's environment.
- (fix): a `prompt_sha256` either benchmark command reports must be the materialized prompt's
  sha256, whatever its type, or the unit is void on the harness with the reason.
- (feat): `icil-orchestrator duel --track T --challenger repo@revision [--size S] --store DIR
  --run-dir DIR` and `icil-orchestrator daemon --store DIR --run-dir DIR [--queue DIR] [--once]`,
  with `--runtime docker|local`, `--local REPO=DIR`, `--live-url` and `--live-token-env`, and
  `--mirror`.
- (feat): `Queue.take` takes an entry off the queue and marks its duel in progress in one write.
- (test): the fake benchmark writes a prompt of named arrays and drives the served policy through
  `RemotePolicy`, so the replay example wins and the zero example loses.
- (chore): scaffold the orchestrator: package, pure test suite, CI and the repository's rules.
