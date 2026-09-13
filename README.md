# ICIL competition orchestrator

The orchestration layer of the RoboTensor one-demonstration in-context imitation learning (ICIL)
competition. It queues submissions, runs duels between a challenger and the reigning king, scores
them, publishes a signed result store and streams live progress to the dashboard.

It contains no benchmark. Benchmarks are separate repositories plugged in through the
`icil.benchmarks` entry point group; the first is
[ICIL-robotwin-benchmark](https://github.com/robotensor/ICIL-robotwin-benchmark). A submission is a
HuggingFace repository with runnable policy code and weights, run in a sandboxed container with no
network.

**Status:** in progress. The first milestone plugs RoboTwin and launches a 1-arm Franka competition
with one sensorimotor demonstration per episode. The contract, benchmark discovery, the signed
store, the queue and live frames are in place; duels and the policy sandbox are not yet.

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"

icil-orchestrator benchmarks list              # declared and installed benchmarks, no import
icil-orchestrator benchmarks check robotwin    # pin, ABI, catalogue, derivation, command builders

icil-orchestrator store init store/            # signing key in keys/ (generated if absent)
icil-orchestrator store verify store/          # signatures, sequence, events, media, schema
icil-orchestrator store mirror store/ --repo owner/dataset

icil-orchestrator queue --store store/ add owner/policy <commit-sha> --duel-size smoke
icil-orchestrator queue list
```

`tests/fixtures/store` is a small signed history for rendering the dashboard
(`ICIL_STORE=$PWD/tests/fixtures/store npm run dev` in the dashboard); it is not a result.
