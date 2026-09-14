"""A duel: both sides of it on the same units and the same prompt bytes, scored and published.

- `runtime`: where a submission's policy runs, as the duel sees it (`docker_runtime` is the
  sandbox, `local_runtime` a subprocess for development).
- `materialize`: every unit's prompt, produced once, before either side runs.
- `side`: one side's units, one served policy per unit, resumable from its results file.
- `score`: per-skill rates, the crown rule and void units.
- `orchestrate`: all of it, from a request to a published record.
"""
