# Changelog

## Unreleased

- (feat): `packages/icil-policy`, the policy protocol: a competitor's policy named by `icil.yaml`
  and served by `python -m icil_policy.serve` to one client, which a benchmark drives with
  `icil_policy.client.RemotePolicy`. Named arrays and JSON fields over an authenticated socket, no
  pickling, object dtypes refused at both ends, hard per-call timeouts, and one
  `PolicyUnavailable` for every failure, `close` apart, which is best effort. Both ends bound what
  a hostile message can cost them: at most 1024 arrays and 32 dimensions each, keys of at least 16
  bytes, an `icil.yaml` of at most 1 MiB whose problems quote only excerpts, and a session that
  ends after `--idle-timeout-s` of silence. Examples `replay_policy` and `zero_policy` are
  complete competitor repositories. Depends on numpy and PyYAML.
- (chore): scaffold the orchestrator: package, pure test suite, CI and the repository's rules.
