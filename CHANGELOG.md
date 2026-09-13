# Changelog

## Unreleased

- (feat): `packages/icil-policy`, the policy protocol: a competitor's policy named by `icil.yaml`
  and served by `python -m icil_policy.serve` to one client, which a benchmark drives with
  `icil_policy.client.RemotePolicy`. Named arrays and JSON fields over an authenticated socket, no
  pickling, object dtypes refused at both ends, hard per-call timeouts, and one
  `PolicyUnavailable` for every failure. Examples `replay_policy` and `zero_policy` are complete
  competitor repositories. Depends on numpy and PyYAML.
- (chore): scaffold the orchestrator: package, pure test suite, CI and the repository's rules.
