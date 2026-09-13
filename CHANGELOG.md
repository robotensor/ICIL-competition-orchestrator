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
- (test): a fake benchmark distribution under `tests/fake_benchmark` and a reproducible fixture
  store under `tests/fixtures/store`.
- (chore): scaffold the orchestrator: package, pure test suite, CI and the repository's rules.
