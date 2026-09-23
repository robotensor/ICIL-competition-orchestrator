# bpp-runtime

The validator's own runtime for BPP (Behavior Prompting Policy) submissions. A miner submits
**weights only** - one `model.safetensors`, no code, no pickle - for the pinned architecture
`bpp_robotwin_l1_v1`. The validator checks the file's header against the template, builds the
network from the template with its own code, loads the weights and serves the policy to the
RoboTwin benchmark over the `vector-policy` socket protocol.

```
src/bpp_runtime/
  check.py, header.py   the file check: header only, standard library only (no torch)
  template.py           the pinned template: arch/bpp_robotwin_l1_v1.{cfg,tensors}.json
  model.py              build the network from the template, load and validate the weights
  demo.py               the benchmark's arrays -> the XPolicyLab form BPP was trained on
  policy.py             BPPPolicy, the vector-policy Policy
  convert.py            a training checkpoint -> model.safetensors (the only unpickling code)
  parity.py             converted weights act exactly as the original checkpoint
third_party/behavior_prompting/   the model source the checkpoints need, vendored (MIT)
```

## Install

```bash
# the validator host: the check only, no dependencies
pip install -e packages/bpp-runtime

# the policy environment (Python 3.12, CUDA): the model, its vendored source, the protocol
pip install -e "packages/bpp-runtime[model]" \
    -e packages/bpp-runtime/third_party/behavior_prompting -e packages/vector-policy
```

## Commands

Each prints one JSON report and exits 0 when it passed, 1 when it did not.

```bash
bpp-runtime check --weights DIR_OR_FILE            # the template's tensors exactly; weights_sha256
bpp-runtime convert --ckpt epoch=0004.ckpt --out DIR   # DIR/model.safetensors, then checked
bpp-runtime template --ckpt epoch0000.ckpt --out src/bpp_runtime/arch   # once, by the organizer
bpp-runtime parity --ckpt X.ckpt --weights DIR --prompt prompt.npz [--seed 0] [--steps 30]
```

`convert` unpickles the checkpoint: run it only on a checkpoint you made.

## Serving

```yaml
# icil.yaml
api: 1
policy: bpp_runtime.policy:BPPPolicy
kwargs: {weights: /abs/path/model.safetensors, device: "cuda:0"}
```

`python -m vector_policy.serve --manifest icil.yaml ...` then serves it. `BPPPolicy` takes
`weights` (an absolute path to `model.safetensors`, or its directory), `device` (default
`cuda:0`), `template` (an absolute template directory; default the packaged one) and
`weights_sha256` (optional: refuse a file with another hash). `action_type` is `ee`: each `act`
answers one `(16,)` float64 row `[left x y z qw qx qy qz, left gripper, right ..., right gripper]`.
Nothing is downloaded: the CLIP ViT is built without timm's pretrained weights and overwritten by
the strict load.

The protocol order is the benchmark's: `reset(seed)` clears the episode (history, cached actions,
prompt) and seeds the diffusion noise, then `set_demonstration`, then one `act` per executed step.

Measured on an RTX PRO 6000 Blackwell (torch 2.8.0+cu128), the BRL1 level-1 checkpoints:
building and loading takes ~12 s (plus ~2 s of imports; `hello` answered in ~13 s), the loaded
policy holds 2.1 GB of GPU memory (~2.9 GB per process with the CUDA context),
`set_demonstration` takes 0.4-0.7 s, an `act` that predicts ~65 ms (the first of a process
~230 ms) and one that pops a cached action ~1.2 ms. `bpp-runtime parity` on both checkpoints:
prompts, predicted chunks and actions all bit-identical to the original checkpoints.

## Tests

```bash
pytest packages/bpp-runtime -m "not gpu"   # the check and the header parser, no torch
pytest packages/bpp-runtime -m gpu         # convert, load, parity on the BRL1 checkpoints
```
