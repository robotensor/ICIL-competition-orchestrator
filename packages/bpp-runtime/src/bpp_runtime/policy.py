"""`BPPPolicy`: a weights-only BPP submission, served over the `vector-policy` protocol.

    # icil.yaml of the validator's own policy repository
    api: 1
    policy: bpp_runtime.policy:BPPPolicy
    kwargs: {weights: /abs/path/to/model.safetensors, device: "cuda:0"}

It reproduces the reference XPolicyLab adapter (`XPolicyLab/policy/BPP/model.py`) on the
benchmark's arrays, and nothing from the submission but its tensors is used:

- `set_demonstration` turns the demonstration into the training data's form (`bpp_runtime.demo`:
  one frame fewer, JPEG round trip, XPolicyLab names) and builds the prompt with the task's own
  dataset class (`only_prompt=True`), as the reference `build_prompt` does. The network encodes it
  once; it is kept until the next `reset`.
- `act` is called once per executed step. Each observation is appended to the history (its length
  from the template; the episode start is padded by repeating the first frame); when the cached
  chunk is empty the network predicts `action_horizon` actions and the first
  `exec_action_horizon` are cached; one is popped and answered as a (16,) `ee` row
  `[left pose 7, left gripper, right pose 7, right gripper]`. That is XPolicyLab's cadence:
  `update_obs` every step, `get_action` when the executed chunk is used up.
- The only randomness is the diffusion sampler's initial noise, drawn from a `torch.Generator`
  seeded by `reset(seed)`: an episode's actions follow from its seed, demonstration and
  observations.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import ARCHITECTURE
from . import demo as demo_mod

log = logging.getLogger("bpp_runtime.policy")

#: The task name the prompt's one-episode replay buffer is filed under (the reference's default).
PROMPT_TASK = "demo"
#: `reset` seeds are reduced into the range a torch generator takes.
SEED_MODULUS = 1 << 64


def mapping_module() -> Any:
    """behavior_prompting's RoboTwin mapping for these checkpoints (UMI-style layout)."""
    from behavior_prompting.train_network.utils import robotwin_bimanual

    return robotwin_bimanual


class BPPPolicy:
    """BPP with weights from `model.safetensors`, built from the validator's pinned template."""

    action_type = "ee"

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        template: str | None = None,
        weights_sha256: str | None = None,
    ) -> None:
        import torch

        from .model import load_policy

        if not os.path.isabs(str(weights)):
            raise ValueError(f"weights must be an absolute path, not {weights!r}")
        if template is not None and not os.path.isabs(str(template)):
            raise ValueError(f"template must be an absolute path, not {template!r}")
        started = time.monotonic()
        self.device = torch.device(device)
        self.weights = str(weights)
        self.network, self.config = load_policy(
            weights,
            device=str(self.device),
            template=template,
            architecture=ARCHITECTURE,
            weights_sha256=weights_sha256,
        )
        self.mapping = mapping_module()
        obs_keys = set(self.config["shape_meta"]["obs"])
        if set(self.mapping.BPP_OBS_KEYS) != obs_keys:
            raise ValueError(
                f"the template's observations {sorted(obs_keys)} are not the mapping's"
            )
        if int(self.config["action_dim"]) != self.mapping.ACTION_DIM:
            raise ValueError(f"the template's action is not {self.mapping.ACTION_DIM}-dimensional")
        self.obs_horizons = {k: int(v) for k, v in self.config["obs_horizons"].items()}
        self.n_obs_steps = int(self.config["n_obs_steps"])
        self.action_horizon = int(self.config["action_horizon"])
        self.exec_action_horizon = int(self.config["exec_action_horizon"])
        if self.network.kwargs:
            raise ValueError(f"the network passes {sorted(self.network.kwargs)} to its sampler")
        self._generator: Any = None
        self._prompt: Any = None
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)
        self._chunk: deque[np.ndarray] = deque()
        #: The last predicted chunk, (action_horizon, 20), for diagnostics and parity.
        self.last_prediction: np.ndarray | None = None
        self.predictions = 0
        self.load_seconds = time.monotonic() - started
        log.info("loaded %s on %s in %.1fs", self.weights, self.device, self.load_seconds)

    # -- the protocol -----------------------------------------------------------------------

    def reset(self, seed: int) -> None:
        """A new episode: no history, no cached actions, no prompt, noise seeded by `seed`."""
        import torch

        self._clear_episode()
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed) % SEED_MODULUS)
        # DiffusionUnetPolicy.predict_action passes its `kwargs` to conditional_sample, whose
        # `generator` feeds torch.randn for the initial noise (and the DDIM step, which draws none).
        self.network.kwargs["generator"] = self._generator

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        """Encode the one demonstration as the network's prompt, kept until the next `reset`."""
        dims = info.get("action_dims") if isinstance(info, Mapping) else None
        if isinstance(dims, Mapping) and dims.get("ee") not in (None, demo_mod.ACTION_DIM):
            raise ValueError(
                f"the robot takes {dims.get('ee')}-wide ee actions; BPP answers "
                f"{demo_mod.ACTION_DIM} (two Franka arms)"
            )
        trajectory = demo_mod.demonstration_trajectory(arrays)
        prompt = self.build_prompt(trajectory)
        self._history.clear()
        self._chunk.clear()
        self.network.reset()
        self.network.prompt(prompt)
        self._prompt = prompt

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """One observation in, one (16,) `ee` action out."""
        if self._generator is None:
            raise RuntimeError("act() before reset()")
        if self._prompt is None:
            raise RuntimeError("act() before set_demonstration()")
        obs = demo_mod.observation(observation)
        self._history.append(self.mapping.xpolicylab_obs_to_bpp_frame(obs))
        if not self._chunk:
            self._chunk.extend(self.predict()[: self.exec_action_horizon])
        action = self.mapping.bpp_action_to_xpolicylab(self._chunk.popleft())
        return {"action": demo_mod.action_row(action)}

    def close(self) -> None:
        """Forget the episode; the network stays loaded."""
        self._clear_episode()
        self._generator = None
        self.network.kwargs.pop("generator", None)

    # -- the network ------------------------------------------------------------------------

    def build_prompt(self, trajectory: Any) -> dict[str, Any]:
        """A demonstration -> the network's prompt tensors (batch 1, on the device), through the
        training dataset's own prompt sampler, exactly as the reference `build_prompt`."""
        import hydra
        import torch
        from behavior_prompting.common.pytorch_util import dict_apply
        from behavior_prompting.common.replay_buffer import ReplayBuffer
        from omegaconf import OmegaConf

        mapping = self.mapping
        traj = mapping.demonstration_to_trajectory(trajectory)
        data = {**mapping.trajectory_lowdim(traj), **mapping.trajectory_images(traj)}
        length = len(data[getattr(mapping, "ACTION_KEY", "action")])
        buffer = ReplayBuffer.create_empty_numpy()
        buffer.add_episode(
            data,
            tasks=[{"name": PROMPT_TASK, "start_idx": 0, "end_idx": length, "labels": {}}],
            episode_name=f"{PROMPT_TASK}/demonstration",
        )
        dataset = hydra.utils.instantiate(
            self.config["dataset"],
            shape_meta=OmegaConf.create(self.config["shape_meta"]),
            replay_buffer=buffer,
            only_prompt=True,
        )
        if len(dataset) < 1:
            raise ValueError("the demonstration yields no prompt")
        prompt = dataset[0]["obs"]["prompt"]
        capacity = int(self.network.obs_encoder.prompt_pos_emb.shape[1])
        if int(prompt["action"].shape[0]) > capacity:
            raise ValueError(
                f"the demonstration is {length} steps, {prompt['action'].shape[0]} prompt chunks; "
                f"the network takes at most {capacity}"
            )
        return dict_apply(
            prompt, lambda x: torch.as_tensor(np.asarray(x)).unsqueeze(0).to(self.device)
        )

    def observation_tensors(self) -> dict[str, Any]:
        """The history as the network's observation, as the reference `_obs_tensor`: the last
        `horizon` frames per key, the episode start padded by repeating the first frame."""
        import torch

        frames = list(self._history)
        frames = [frames[0]] * (self.n_obs_steps - len(frames)) + frames
        out = {}
        for key, horizon in self.obs_horizons.items():
            stack = np.stack([frame[key] for frame in frames[-horizon:]])
            if key in self.mapping.BPP_RGB_KEYS:
                stack = np.moveaxis(stack, -1, 1).astype(np.float32) / 255.0
            out[key] = torch.from_numpy(stack.astype(np.float32)[None]).to(self.device)
        return out

    def predict(self) -> np.ndarray:
        """One `predict_action` on the history: (action_horizon, 20) float32."""
        import torch

        with torch.inference_mode():
            result = self.network.predict_action(self.observation_tensors())
        chunk = result["action"].float().cpu().numpy()[0]
        self.last_prediction = chunk
        self.predictions += 1
        return chunk

    def _clear_episode(self) -> None:
        self._history.clear()
        self._chunk.clear()
        self._prompt = None
        self.last_prediction = None
        self.network.reset()
