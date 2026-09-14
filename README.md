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
store, the queue, live frames, the policy protocol and the policy sandbox are in place; duels are
not yet.

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]" -e packages/icil-policy

icil-orchestrator benchmarks list              # declared and installed benchmarks, no import
icil-orchestrator benchmarks check robotwin    # pin, ABI, catalogue, derivation, command builders

icil-orchestrator store init store/            # signing key in keys/ (generated if absent)
icil-orchestrator store verify store/ --validator-key <hex>   # signatures, events, media, schema
icil-orchestrator store mirror store/ --repo owner/dataset

icil-orchestrator queue --store store/ add owner/policy main --duel-size smoke   # resolved to its commit
icil-orchestrator queue list

icil-orchestrator submission build-base                  # docker/policy-base, prints its digest
icil-orchestrator submission check owner/policy@main --base-image sha256:<hex>   # resolve, fetch,
                                                         # manifest, build, run, hello; reported
icil-orchestrator submission check local/replay@main --local packages/icil-policy/examples/replay_policy
icil-orchestrator submission prune                       # the icil-submission images nothing uses
```

A submission is a Hugging Face repository at a commit: `queue add` resolves a branch or a tag to
its sha once, through the Hub, confirms a sha it is given, and everything published hangs off that
sha. `submission check` fetches it into `cache/<sha>/` (git-ignored), reads its `icil.yaml` as a
plain file, builds its image `FROM` the pinned base by digest with its checkout copied in and its
requirements installed at build time (bounded by `--build-timeout`), runs it under
`spec.submission.sandbox` - no network, a read-only root, `/tmp` as tmpfs, a non-root user, memory
with no swap, cpu and pid limits, one directory shared for the socket (a small tmpfs when the
orchestrator is root) - and says `hello`. A step that fails is the submission's rejection with the
reason or the harness's error, and the container is removed either way; a container whose process
was killed is removed by the next start. `--local DIR` takes a directory in the Hub's place. See
[`docker/policy-base`](docker/policy-base/README.md) for the base image, what a policy finds at
run time and what was measured.

`store init` writes the store's ed25519 signing key to `keys/orchestrator.ed25519` (mode 0600)
unless `--key` says otherwise. It is the only thing that can publish as this store, so keep it out
of the checkout and off the mirror: `/keys/` and `/queue/` are git-ignored, and the mirror uploads
the store's layout and nothing beside it.

`tests/fixtures/store` is a small signed history for rendering the dashboard
(`ICIL_STORE=$PWD/tests/fixtures/store npm run dev` in the dashboard); it is not a result.
