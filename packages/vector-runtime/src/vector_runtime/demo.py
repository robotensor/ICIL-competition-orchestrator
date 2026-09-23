"""The benchmark's arrays, as the data BPP was trained on: numpy and OpenCV, no torch.

BPP learned from RoboTwin episodes written by `envs/utils/pkl2hdf5.py` in the XPolicyLab HDF5
layout (data_format v1.0, `dual_franka`), so a demonstration is made to look exactly like one:

- **one frame fewer**: HDF5 `state` is the recording's rows `[:-1]`, `action` its rows `[1:]`,
  and the camera frames are `[:-1]`;
- **JPEG**: every stored camera frame went through `images_encoding` -> `encode_image_bit`
  (`cv2.imencode(".jpg", RGB->BGR)`, OpenCV's default quality) and back through
  `decode_image_bit` (`cv2.imdecode`, BGR->RGB). The RGB marker `encode_image_bit` adds is a
  JPEG comment and changes no pixel;
- **names**: RoboTwin's `head_camera`, `left_camera`, `right_camera` are XPolicyLab's `cam_head`,
  `cam_left_wrist`, `cam_right_wrist`; per arm, the endpose row's pose is `{arm}_ee_pose`
  `[x, y, z, qw, qx, qy, qz]` and its gripper `{arm}_ee_joint_state`.

A live observation is converted the same way but without JPEG: XPolicyLab hands a policy the raw
RGB frame. Everything after this - resizing, poses in BPP's representation, the prompt - is
behavior_prompting's own `train_network/utils/robotwin_bimanual.py` and dataset class.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

#: RoboTwin camera -> XPolicyLab camera.
CAMERAS = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
FRAMES_PREFIX = "frames_"
ARMS = ("left", "right")
POSE_DIM = 7
#: The benchmark's `endpose` row: per arm, left then right, pose 7 then gripper 1.
ENDPOSE_DIM = len(ARMS) * (POSE_DIM + 1)
#: The `ee` action row BPPPolicy answers with, in the same layout.
ACTION_DIM = ENDPOSE_DIM


class DemonstrationError(ValueError):
    """The arrays are not a demonstration or an observation this runtime can use."""


def jpeg_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """One RGB frame as XPolicyLab stores and reloads it: `encode_image_bit`, `decode_image_bit`."""
    import cv2

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise DemonstrationError(f"cannot JPEG-encode a frame of shape {rgb.shape}")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _frames(arrays: Mapping[str, Any], camera: str, ndim: int) -> np.ndarray:
    name = FRAMES_PREFIX + camera
    if name not in arrays:
        raise DemonstrationError(f"no {name!r} array; BPP needs {sorted(CAMERAS)}")
    frames = np.asarray(arrays[name])
    if frames.dtype != np.uint8 or frames.ndim != ndim or frames.shape[-1] != 3:
        want = "(T, H, W, 3)" if ndim == 4 else "(H, W, 3)"
        raise DemonstrationError(f"{name} is {frames.dtype} {frames.shape}, expected uint8 {want}")
    return frames


def _endpose(arrays: Mapping[str, Any], ndim: int) -> np.ndarray:
    if "endpose" not in arrays:
        raise DemonstrationError("no 'endpose' array")
    endpose = np.asarray(arrays["endpose"])
    if endpose.ndim != ndim or endpose.shape[-1] != ENDPOSE_DIM or endpose.dtype.kind not in "fiu":
        raise DemonstrationError(f"endpose is {endpose.dtype} {endpose.shape}, expected (..., 16)")
    endpose = endpose.astype(np.float64)
    if not np.all(np.isfinite(endpose)):
        raise DemonstrationError("endpose holds non-finite values")
    return endpose


def _arm(rows: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    start = ARMS.index(arm) * (POSE_DIM + 1)
    return rows[..., start : start + POSE_DIM], rows[..., start + POSE_DIM : start + POSE_DIM + 1]


def demonstration_trajectory(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """The benchmark's demonstration arrays as the XPolicyLab trajectory training read.

    `{"state": {...}, "action": {...}, "vision": {cam: {"color": (T-1, H, W, 3)}}}`, in the form
    `robotwin_bimanual.demonstration_to_trajectory` takes.
    """
    endpose = _endpose(arrays, 2)
    count = len(endpose)
    if count < 2:
        raise DemonstrationError(f"a demonstration needs at least 2 frames, not {count}")
    frames = {camera: _frames(arrays, camera, 4) for camera in CAMERAS}
    for camera, block in frames.items():
        if len(block) != count:
            raise DemonstrationError(f"{camera} has {len(block)} frames; endpose has {count}")

    trajectory: dict[str, Any] = {"state": {}, "action": {}, "vision": {}}
    for arm in ARMS:
        pose, gripper = _arm(endpose, arm)
        trajectory["state"][f"{arm}_ee_pose"] = pose[:-1]
        trajectory["state"][f"{arm}_ee_joint_state"] = gripper[:-1]
        trajectory["action"][f"{arm}_ee_pose"] = pose[1:]
        trajectory["action"][f"{arm}_ee_joint_state"] = gripper[1:]
    for camera, name in CAMERAS.items():
        stored = np.stack([jpeg_roundtrip(frame) for frame in frames[camera][:-1]])
        trajectory["vision"][name] = {"color": stored}
    return trajectory


def observation(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """One benchmark observation as the live XPolicyLab observation a deployed policy gets:
    `{"vision": {cam: {"color": (H, W, 3) RGB}}, "state": {...}}`, frames untouched."""
    endpose = _endpose(arrays, 1)
    obs: dict[str, Any] = {"vision": {}, "state": {}}
    for camera, name in CAMERAS.items():
        obs["vision"][name] = {"color": _frames(arrays, camera, 3)}
    for arm in ARMS:
        pose, gripper = _arm(endpose, arm)
        obs["state"][f"{arm}_ee_pose"] = pose
        obs["state"][f"{arm}_ee_joint_state"] = gripper
    return obs


def action_row(action: Mapping[str, Any]) -> np.ndarray:
    """An XPolicyLab `ee` action dict as the benchmark's (16,) float64 row."""
    parts = []
    for arm in ARMS:
        parts.append(np.asarray(action[f"{arm}_ee_pose"], dtype=np.float64).reshape(POSE_DIM))
        parts.append(np.asarray(action[f"{arm}_ee_joint_state"], dtype=np.float64).reshape(1))
    return np.concatenate(parts)
