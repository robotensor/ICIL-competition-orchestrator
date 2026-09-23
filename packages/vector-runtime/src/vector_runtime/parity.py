"""Converted weights served by `BPPPolicy` act exactly as the original checkpoint does.

    vector-runtime parity --ckpt X.ckpt --weights DIR --prompt PROMPT.npz [--seed 0] [--steps 30]
                       [--xpolicylab DIR] [--tolerance 1e-5]

Two policies, one process, one GPU:

- **the reference**: the checkpoint loaded the way XPolicyLab's BPP adapter
  (`XPolicyLab/policy/BPP/model.py`) loads it - `torch.load` with dill, the model instantiated from
  the checkpoint's own configuration (timm fetches the CLIP weights, as in training) after
  `TrainPolicyWorkspace`'s seeding, `load_state_dict` of `state_dicts.model`, the prompt from the
  checkpoint's own `task.dataset` - and driven with the adapter's own steps (`build_prompt`,
  `update_obs` / `_obs_tensor` / `get_action`, `exec_action_horizon` actions per prediction);
- **the candidate**: `BPPPolicy` on the converted `model.safetensors` and the pinned template.

Both get the prompt file's demonstration and then its own frames and endpose rows as successive
observations, with diffusion noise drawn from a generator seeded alike, and every action must
match. When XPolicyLab is importable (`--xpolicylab`, or already on the path) the reference is
handed the demonstration as a real XPolicyLab HDF5 file, written as RoboTwin's `pkl2hdf5` writes
one (`images_encoding`, fixed-width byte columns) and read back by behavior_prompting's own
reader, so the candidate's in-memory conversion (`vector_runtime.demo`) is checked too. Otherwise
the reference gets the candidate's converted trajectory and only the model is compared.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from . import demo as demo_mod
from .template import DEFAULT_EXEC_ACTION_HORIZON

PRIVILEGED = "meta"


def read_prompt(path: str | os.PathLike[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The policy's arrays of a benchmark prompt file (never `meta`) and a public `info`."""
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files if name != PRIVILEGED}
    cameras = sorted(
        name[len(demo_mod.FRAMES_PREFIX) :]
        for name in arrays
        if name.startswith(demo_mod.FRAMES_PREFIX)
    )
    info = {
        "frequency": float(arrays["frequency"]),
        "cameras": cameras,
        "embodiment": "franka-panda",
        "action_dims": {"qpos": int(arrays["qpos"].shape[1]), "ee": demo_mod.ENDPOSE_DIM},
    }
    return arrays, info


class Reference:
    """The checkpoint, loaded and driven the way the XPolicyLab BPP adapter does it."""

    def __init__(self, ckpt: str | os.PathLike[str], device: str, exec_action_horizon: int):
        import hydra
        import torch

        from .convert import load_checkpoint
        from .policy import mapping_module

        payload = load_checkpoint(ckpt)
        self.cfg = payload["cfg"]
        # TrainPolicyWorkspace.__init__: seed everything, then instantiate the model.
        seed = self.cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.policy = hydra.utils.instantiate(self.cfg.model)
        # BaseWorkspace.load_payload(payload, exclude_keys=["optimizer"]): the model's state dict.
        self.policy.load_state_dict(payload["state_dicts"]["model"])
        del payload
        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.shape_meta = self.cfg.task.shape_meta
        self.mapping = mapping_module()
        self.obs_horizons = {k: int(v.horizon) for k, v in self.shape_meta.obs.items()}
        self.n_obs_steps = max(self.obs_horizons.values())
        self.exec_action_horizon = int(exec_action_horizon)
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)
        self.prompt_tensors: dict[str, Any] | None = None
        self.last_prediction: np.ndarray | None = None

    def build_prompt(self, demo: Any, task_name: str = "demo") -> dict[str, Any]:
        import hydra
        import omegaconf
        from behavior_prompting.common.replay_buffer import ReplayBuffer

        traj = self.mapping.demonstration_to_trajectory(demo)
        data = {**self.mapping.trajectory_lowdim(traj), **self.mapping.trajectory_images(traj)}
        length = len(data[getattr(self.mapping, "ACTION_KEY", "action")])
        rb = ReplayBuffer.create_empty_numpy()
        rb.add_episode(
            data,
            tasks=[{"name": task_name, "start_idx": 0, "end_idx": length, "labels": {}}],
            episode_name=f"{task_name}/demonstration",
        )
        dataset_cfg = omegaconf.OmegaConf.to_container(self.cfg.task.dataset, resolve=True)
        dataset_cfg.update(dataset_path=None, cache_dir=None, val_ratio=0.0)
        dataset = hydra.utils.instantiate(
            dataset_cfg, shape_meta=self.shape_meta, replay_buffer=rb, only_prompt=True
        )
        assert len(dataset) >= 1
        return dataset[0]["obs"]["prompt"]

    def set_demonstration(self, demo: Any) -> None:
        import torch
        from behavior_prompting.common.pytorch_util import dict_apply

        prompt = self.build_prompt(demo)
        self.prompt_tensors = dict_apply(
            prompt, lambda x: torch.as_tensor(np.asarray(x)).unsqueeze(0).to(self.device)
        )
        self.policy.reset()
        self.policy.prompt(self.prompt_tensors)

    def reset(self, seed: int) -> None:
        import torch

        self.history.clear()
        self.policy.reset()
        self.policy.prompt(self.prompt_tensors)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed) % (1 << 64))
        self.policy.kwargs["generator"] = generator

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.history.append(self.mapping.xpolicylab_obs_to_bpp_frame(obs))

    def get_action(self) -> list[dict[str, np.ndarray]]:
        import torch

        frames = list(self.history)
        frames = [frames[0]] * (self.n_obs_steps - len(frames)) + frames
        per_env = {}
        for key, horizon in self.obs_horizons.items():
            stack = np.stack([f[key] for f in frames[-horizon:]])
            if key in self.mapping.BPP_RGB_KEYS:
                stack = np.moveaxis(stack, -1, 1).astype(np.float32) / 255.0
            per_env[key] = stack.astype(np.float32)
        obs = {k: torch.from_numpy(np.stack([v])).to(self.device) for k, v in per_env.items()}
        with torch.inference_mode():
            actions = self.policy.predict_action(obs)["action"].float().cpu().numpy()
        self.last_prediction = actions[0]
        return [
            self.mapping.bpp_action_to_xpolicylab(a) for a in actions[0][: self.exec_action_horizon]
        ]


def xpolicylab_observation(arrays: dict[str, np.ndarray], row: int) -> dict[str, Any]:
    """Row `row` of a prompt as the live observation XPolicyLab would hand the adapter."""
    endpose = np.asarray(arrays["endpose"][row], dtype=np.float64)
    obs: dict[str, Any] = {"vision": {}, "state": {}}
    for camera, name in demo_mod.CAMERAS.items():
        obs["vision"][name] = {"color": arrays[demo_mod.FRAMES_PREFIX + camera][row]}
    for k, arm in enumerate(demo_mod.ARMS):
        start = k * (demo_mod.POSE_DIM + 1)
        obs["state"][f"{arm}_ee_pose"] = endpose[start : start + demo_mod.POSE_DIM]
        obs["state"][f"{arm}_ee_joint_state"] = endpose[start + demo_mod.POSE_DIM : start + 8]
    return obs


def write_xpolicylab_hdf5(arrays: dict[str, np.ndarray], path: Path) -> None:
    """The demonstration as RoboTwin's `pkl2hdf5.create_xpolicylab_hdf5` writes an episode."""
    import h5py
    from XPolicyLab.utils.process_data import images_encoding

    endpose = np.asarray(arrays["endpose"], dtype=np.float64)
    with h5py.File(path, "w") as f:
        state, action, vision = (
            f.create_group("state"),
            f.create_group("action"),
            f.create_group("vision"),
        )
        for k, arm in enumerate(demo_mod.ARMS):
            start = k * (demo_mod.POSE_DIM + 1)
            pose = endpose[:, start : start + demo_mod.POSE_DIM]
            gripper = endpose[:, start + demo_mod.POSE_DIM : start + demo_mod.POSE_DIM + 1]
            state.create_dataset(f"{arm}_ee_poses", data=pose[:-1])
            action.create_dataset(f"{arm}_ee_poses", data=pose[1:])
            state.create_dataset(f"{arm}_ee_joint_states", data=gripper[:-1])
            action.create_dataset(f"{arm}_ee_joint_states", data=gripper[1:])
        for camera, name in demo_mod.CAMERAS.items():
            colors = np.asarray(arrays[demo_mod.FRAMES_PREFIX + camera])[:-1]
            encoded, max_len = images_encoding(colors)
            vision.create_group(name).create_dataset("colors", data=encoded, dtype=f"S{max_len}")


def _xpolicylab_available(root: str | None) -> bool:
    if root:
        sys.path.insert(0, str(root))
    try:
        import XPolicyLab.utils.process_data  # noqa: F401
    except ImportError:
        return False
    return True


def run(
    ckpt: str | os.PathLike[str],
    weights: str | os.PathLike[str],
    prompt: str | os.PathLike[str],
    *,
    seed: int = 0,
    steps: int = 30,
    device: str = "cuda:0",
    xpolicylab: str | None = None,
    tolerance: float = 1e-5,
    template: str | None = None,
) -> dict[str, Any]:
    """Drive both policies through `steps` observations; the comparison, as a report."""
    import torch

    from .check import weights_file
    from .policy import BPPPolicy

    arrays, info = read_prompt(prompt)
    exec_horizon = DEFAULT_EXEC_ACTION_HORIZON
    candidate = BPPPolicy(
        weights=str(weights_file(Path(weights).absolute())), device=device, template=template
    )
    if candidate.exec_action_horizon != exec_horizon:
        exec_horizon = candidate.exec_action_horizon
    reference = Reference(ckpt, device, exec_horizon)

    via_hdf5 = _xpolicylab_available(xpolicylab)
    candidate.reset(seed)
    candidate.set_demonstration(arrays, info)
    if via_hdf5:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_0000000.hdf5"
            write_xpolicylab_hdf5(arrays, path)
            reference.set_demonstration(str(path))
    else:
        reference.set_demonstration(demo_mod.demonstration_trajectory(arrays))
    reference.reset(seed)

    prompt_diff = _max_diff(candidate._prompt, reference.prompt_tensors)
    queue: deque[dict[str, np.ndarray]] = deque()
    action_diff = chunk_diff = 0.0
    predictions = 0
    count = len(arrays["endpose"])
    first_actions = []
    for step in range(steps):
        row = min(step, count - 1)
        observation = {
            name: arrays[name][row]
            for name in arrays
            if name.startswith(demo_mod.FRAMES_PREFIX) or name in ("qpos", "endpose")
        }
        before = candidate.predictions
        mine = candidate.act(observation)["action"]
        reference.update_obs(xpolicylab_observation(arrays, row))
        if not queue:
            queue.extend(reference.get_action())
            predictions += 1
        theirs = demo_mod.action_row(queue.popleft())
        if candidate.predictions != before:
            chunk_diff = max(
                chunk_diff,
                float(np.abs(candidate.last_prediction - reference.last_prediction).max()),
            )
        action_diff = max(action_diff, float(np.abs(mine - theirs).max()))
        if step < 2:
            first_actions.append(mine.tolist())
    if candidate.predictions != predictions:
        raise RuntimeError(
            f"the candidate predicted {candidate.predictions} times, the reference {predictions}"
        )
    report = {
        "ok": action_diff <= tolerance and chunk_diff <= tolerance,
        "exact": action_diff == 0.0 and chunk_diff == 0.0 and prompt_diff == 0.0,
        "seed": seed,
        "steps": steps,
        "predictions": predictions,
        "exec_action_horizon": exec_horizon,
        "demonstration_frames": count,
        "reference_demonstration": "xpolicylab-hdf5" if via_hdf5 else "shared-trajectory",
        "max_abs_diff_prompt": prompt_diff,
        "max_abs_diff_chunk": chunk_diff,
        "max_abs_diff_action": action_diff,
        "tolerance": tolerance,
        "first_actions": first_actions,
        "device": torch.cuda.get_device_name(torch.device(device)) if "cuda" in device else device,
    }
    return report


def _max_diff(a: Any, b: Any) -> float:
    """The largest absolute difference between two nested dicts of tensors of equal structure."""
    import torch

    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise RuntimeError(
                f"prompt structures differ: {sorted(a)} vs {sorted(b) if isinstance(b, dict) else b}"
            )
        return max((_max_diff(a[k], b[k]) for k in a), default=0.0)
    if tuple(a.shape) != tuple(b.shape):
        raise RuntimeError(f"prompt shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    if a.dtype == torch.bool:
        return float((a != b).any())
    return float((a.double() - b.double()).abs().max())
