# Changelog

## Unreleased

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
  carries a build budget) and one past it is a rejection; a build whose pip could not reach its
  index is the harness's error, not a rejection. The session kept after `hello` is bounded by
  `act_timeout_s`. A file the Hub declares no size for is not downloaded; a missing branch or
  repository is rejected with the Hub's words for it; `queue add` confirms a commit sha on the Hub
  too; `--local` holds the ref to a repo id. Containers are labelled with the process that started
  them, and a start reaps those whose process has ended, with their tmpfs.
- (feat): `icil-orchestrator submission prune` removes the `icil-submission` images no container
  uses. The checkout is copied into its image rather than bind-mounted read-only as issue #4's
  scope put it: the requirements install needs it at build time, the recorded image id is the code
  that ran, and the container mounts nothing of the host but its socket directory.
- (chore): scaffold the orchestrator: package, pure test suite, CI and the repository's rules.
