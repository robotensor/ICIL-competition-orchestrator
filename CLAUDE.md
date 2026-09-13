# icil-orchestrator — the ICIL competition's orchestration layer

Python 3.10, package `icil_orchestrator` under `src/`. This repository runs the RoboTensor
one-demonstration in-context imitation learning competition: the challenger queue, duels, scoring,
the signed result store and live progress for the dashboard. It contains **no benchmark**.
Benchmarks are separate distributions plugged in through the `icil.benchmarks` entry point group;
the first is `robotensor/ICIL-robotwin-benchmark`. Submissions are HuggingFace repositories holding
runnable policy code and weights, run in a sandboxed container.

## Where the pieces come from

- The orchestration core is ported from `robotensor/ICIL-LiberoGen-bench` branch
  `milestone-two-fields` (`src/icilval/`): `benchmarks/{api,subprocess_runner,units}.py`,
  `materialize.py`, `model/{wire,host}.py`, `store/`, `queue.py`, `daemon.py`, `live.py`,
  `duel/`, `ids.py`, `canon.py`, `rng.py`, `spec.py`. Read the original before porting a module,
  and cut what exists only because submissions were weights: `arch.py`,
  `model/{architectures,fingerprint,convert,bpp_robotwin}`, `pools/`, `simulators/`, `demoview.py`.
- `robofluent/ICIL-competition-dashboard` renders this repository's `spec.json`,
  `store-schema.json`, store layout and live frames. A change to any of them is a dashboard change
  too, and needs a dashboard issue.

## Commands

- Host env: `uv venv --python 3.10 .venv && uv pip install -e ".[dev]"`; `ruff check . && ruff format --check .`; `pytest -m "not sim and not container"`.
- Tests run a benchmark through `tests/fake_benchmark` (a real `.dist-info` on `sys.path`; its
  command half is a script). After an intended change to `spec.json` or the store layout,
  regenerate the fixture store with `python tests/fixtures/make_store.py` and commit it.
- `live.PHASES` must equal the dashboard's `PHASES` (`lib/live/types.ts`); a new phase is a
  dashboard change first.

## Rules

- No benchmark here, and no simulator import anywhere in `icil_orchestrator`. A benchmark's pure
  surface is called in-process; everything that needs a simulator runs as the argv its plugin
  returns. A benchmark never imports this package.
- Competitor code is untrusted and runs only inside its policy container: the pinned base image,
  the HF repo at a resolved commit sha, `--network none`, a read-only root filesystem, a non-root
  user and resource limits. The orchestrator never imports, unpickles or executes anything from a
  submission, and a policy container never mounts the store, prompt metadata or the other side's
  files.
- No architecture or model-type check. A submission satisfies `icil.yaml` and the policy protocol:
  it answers `hello`, accepts one demonstration and returns actions of the benchmark's shape in time.
- The wire carries named arrays and JSON fields only. Never pickle; object dtypes are refused at
  both ends.
- Both sides of a duel get identical demonstration bytes: prompts are materialized once per duel,
  before either side runs, and published with the event by sha256. Reproducibility is verification
  by hash, not regeneration — an expert's trajectory is not reproducible from its seed.
- No privileged data reaches a policy: prompt `meta` (scene seed, scene digest, success condition)
  stays on the benchmark side of the socket.
- Everything published is deterministic from `spec.json`, the duel id and the two submission refs:
  unit lists, seeds, ids. No clocks and no global RNG in anything that is published.
- A unit that fails for a harness reason is void, not a loss; `max_void_fraction` decides whether
  the duel stands.
- `spec.json` and `store-schema.json` are the contract. Read numbers through the spec module; never
  repeat one as a literal. Stored scores are fractions in `[0, 1]`.

## Conventions

- Small commits. One concern per commit (a rename, a schema change, a new stage, a doc update),
  never a whole issue in one commit. Each commit builds and passes the pure tests on its own, so
  the history bisects and reverts cleanly. Split mechanical moves from behaviour changes.
- Commit title: `(feat): …`, `(fix): …`, `(refactor): …`, `(docs): …`, `(test): …`, `(chore): …`;
  imperative, lower-case after the prefix, under 72 characters, no trailing period. Body: why the
  change, not what the diff shows; short bullets; `Refs #N` for the issue it advances,
  `Closes #N` only on the commit that finishes it.
- Issues stand on their own: someone who was not in the conversation that produced one must be
  able to act on it. The title is the outcome in plain words - what is true once it closes
  ("Rebuild the identical scene and verify it matches") - not a component name or a plan step; a
  bug's title is its symptom. The body, in this order:
  - `## Why`: the problem, and what goes wrong without the change. No "see the plan", no "as
    discussed".
  - `## Scope`: the deliverable as concrete bullets (behaviour, files, commands), then
    `Out of scope:` for what a reader might expect and will not get.
  - `## Acceptance criteria`: a `- [ ]` checklist of things that can be checked - a test, a
    command and its result, an observable behaviour. Never "works well".
  - `## Notes`, optional: constraints, pitfalls, upstream references with paths, `Depends on #N`.
  - A bug has `## What happened` (the command, the commit, the evidence), `## Expected` and, once
    known, `## Cause`, in place of Why and Scope.
  - On closing, add `## Outcome`: what shipped and in which PRs, the measured result, and anything
    that differs from the scope. A criterion that was dropped or changed is said, not silently
    ticked.
- One issue is one deliverable. Label it with its area, add `bug` for a defect, and put it in a
  milestone when the work belongs to one; split anything that will not land in one go and link the
  parts with `Depends on #N`.
- A branch carries a theme, not an issue number: related issues that touch the same code ship on
  one branch (`short-slug`, or `issue-N-short-slug` when it really is a single issue) and land in
  one PR, which says `Closes #N` for every issue it finishes and `Refs #N` for the ones it only
  advances. Tests and a CHANGELOG entry land with it. Rebase, do not merge `main` into the branch.
- Small changes go straight to `main`: a typo, a comment, a doc line, a version bump, a one-line
  fix that comes with its test. Anything that changes behaviour a reader would need explained,
  touches a published contract, or wants a second pair of eyes takes a branch and a PR.
- When a branch is merged, delete it locally and on the remote, so only `main`, long-lived
  `milestone-*` branches and deliberate `archive/*` refs remain.
- Published results are hosted: the signed store is mirrored to a Hugging Face dataset. What stays a
  plain file in the repository or a run directory is everything not published - run logs, side
  output, local records.
