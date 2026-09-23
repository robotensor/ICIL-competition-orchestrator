# behavior_prompting, vendored

The part of `behavior_prompting` (Behavior Prompting Policy, MIT, (c) 2026 Austin Patel; see
`LICENSE`) that `bpp-runtime` needs to build, load, prompt and run the BPP RoboTwin bimanual
checkpoints. It is the exact source those checkpoints were trained with.

## Origin

- Repository: Hugging Face `louis392/BRL1`, revision `90de4af4d77e559f67c1142d0c4dcbcd3671dcdc`
- File: `code/behavior_prompting_snapshot.tar.gz`,
  sha256 `07a8acccb08e40a4fe5337cf0af5e56e3af19ed0189d0eba63e21771ebbf5e4c`
- Every file here is byte-identical to the file of the same path under the tarball's
  `behavior_prompting/` directory: `sha256sum -c SHA256SUMS` from that directory passes.

## What is kept

The 30 files the runtime imports - found by running the whole path (build the network from the
template, load the weights, build a prompt with `UmiTaskDataset(only_prompt=True)`, predict, and
the parity reference) and listing every `behavior_prompting` module loaded - plus `LICENSE`.
`behavior_prompting/__init__.py` and `train_network/__init__.py` are the only `__init__.py` files
upstream has on this path; the directories below them are namespace packages, as upstream.

Left out: training, workspaces, env runners and rollout code, LIBERO / draw / UMI environments,
their BDDL, init-state and asset files, experiment and Hydra configuration (the runtime uses the
resolved configuration pinned in `bpp_runtime/arch/`), scripts, docs and `__pycache__`.

## Install

```bash
pip install -e packages/bpp-runtime/third_party/behavior_prompting
```

This replaces any other `behavior_prompting` install: only one may be importable.
