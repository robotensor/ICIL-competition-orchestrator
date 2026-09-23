"""Data mapping between RoboTwin XPolicyLab trajectories / observations and BPP, bimanual franka in end-effector space.

Used by both sides so training and deployment can never disagree:
  - train_network/scripts/robotwin/convert_xpolicylab_to_bpp.py   HDF5 trajectories -> BPP zarr replay buffer
  - XPolicyLab/policy/BPP/model.py (RoboTwin repo)                 live observations -> policy frames, predictions -> actions

RoboTwin conventions (XPolicyLab data_format v1.0, dual_franka):
  state/{left,right}_ee_pose[s]         (T, 7) world frame [x, y, z, qw, qx, qy, qz]  (plural in HDF5, singular in live obs)
  state/{left,right}_ee_joint_state[s]  (T, 1) gripper opening in [0, 1]
  action/...                            same keys; action[t] is the target reached at t + 1
  vision/<cam>/colors (HDF5, encoded)   or vision/<cam>/color (live obs, RGB)  for cam_head, cam_left_wrist, cam_right_wrist

BPP side (UMI-style keys, so `UmiTaskDataset` is used unchanged):
  replay buffer   gripper_{left,right}_eef_pos (T, 3), _eef_rot_axis_angle (T, 3, rotvec), _gripper_width (T, 1),
                  action (T, 14) = per arm [pos 3, rotvec 3, gripper 1] from RoboTwin's action stream,
                  camera_{head,left_wrist,right_wrist}_rgb (T, 224, 224, 3) uint8 RGB
  policy obs      absolute poses (pose_repr abs) as pos 3 + rot6d 6, gripper width, and each arm's pose in the other
                  arm's frame. Low-dim horizon is 1, so every value depends on one frame only and a frame can be built
                  from a single live observation.
  action (20)     per arm [pos 3, rot6d 6, gripper 1], absolute world frame, left arm first.
"""
import os

import cv2
import h5py
import numpy as np

from behavior_prompting.common.pose_util import mat_to_pose10d, pose10d_to_mat, pose_to_mat

IMAGE_SIZE = 224
ARMS = ("left", "right")
ROBOT_PREFIXES = {arm: f"gripper_{arm}" for arm in ARMS}
# XPolicyLab camera name -> BPP obs key
CAMERAS = {"cam_head": "camera_head_rgb", "cam_left_wrist": "camera_left_wrist_rgb", "cam_right_wrist": "camera_right_wrist_rgb"}

BPP_RGB_KEYS = list(CAMERAS.values())
BPP_LOWDIM_KEYS = [f"gripper_{arm}_{k}" for arm in ARMS for k in ("eef_pos", "eef_rot_axis_angle", "gripper_width")] + [
    "gripper_left_eef_pos_wrt_gripper_right", "gripper_right_eef_pos_wrt_gripper_left",
    "gripper_left_eef_rot_axis_angle_wrt_gripper_right", "gripper_right_eef_rot_axis_angle_wrt_gripper_left"]
BPP_OBS_KEYS = BPP_RGB_KEYS + BPP_LOWDIM_KEYS
ACTION_DIM = 20
RAW_ACTION_DIM = 14  # replay buffer action: per arm pos 3 + rotvec 3 + gripper 1


# ---------------------------------------------------------------------------------------------- rotations
def quat_wxyz_to_rotvec(quat):
    """(..., 4) wxyz -> (..., 3) axis-angle, through the rotation matrix (sign of the quaternion does not matter)."""
    from scipy.spatial.transform import Rotation
    quat = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_rotvec().reshape(quat.shape[:-1] + (3,))


def mat_to_quat_wxyz(mat):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(mat, dtype=np.float64)).as_quat()  # xyzw
    q = q[..., [3, 0, 1, 2]]
    return np.where(q[..., :1] < 0, -q, q)  # canonical hemisphere (qw >= 0)


def ee_pose_to_mat(pose7):
    """(..., 7) [x, y, z, qw, qx, qy, qz] -> (..., 4, 4)."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    return pose_to_mat(np.concatenate([pose7[..., :3], quat_wxyz_to_rotvec(pose7[..., 3:7])], axis=-1))


# ---------------------------------------------------------------------------------------------- images
def resize_image(image):
    """RGB (H, W, 3) uint8 at any size -> (224, 224, 3) uint8. The one resize used by the converter and at inference."""
    image = np.asarray(image)
    if image.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE):
        return image.astype(np.uint8, copy=False)
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def _decode_frames(frames):
    """Encoded image bits (HDF5 `colors`) or already decoded RGB frames -> (T, H, W, 3) uint8 RGB."""
    frames = np.asarray(frames) if not isinstance(frames, list) else frames
    if isinstance(frames, np.ndarray) and frames.dtype == np.uint8 and frames.ndim == 4:
        return frames
    # always through decode_image_bit: stored bits come in two byte formats and only it tells them apart
    from XPolicyLab.utils.process_data import decode_image_bit
    return np.asarray(decode_image_bit(frames))


# ---------------------------------------------------------------------------------------------- trajectories
def _get(group, *names):
    for name in names:
        if name in group:
            return group[name]
    raise KeyError(f"none of {names} in {list(group.keys())}")


def _load_hdf5(path):
    with h5py.File(path, "r") as f:
        traj = {"state": {}, "action": {}, "vision": {}}
        for arm in ARMS:
            for src in ("state", "action"):
                if src in f:
                    traj[src][f"{arm}_ee_pose"] = np.asarray(_get(f[src], f"{arm}_ee_poses", f"{arm}_ee_pose"), dtype=np.float64)
                    traj[src][f"{arm}_ee_joint_state"] = np.asarray(
                        _get(f[src], f"{arm}_ee_joint_states", f"{arm}_ee_joint_state"), dtype=np.float64).reshape(-1, 1)
        for cam in CAMERAS:
            group = f["vision"][cam]
            traj["vision"][cam] = _decode_frames(_get(group, "colors", "color")[:])
        if not traj["action"]:
            del traj["action"]
    return traj


def _stack_obs_list(observations, actions=None):
    """[live XPolicyLab obs dicts] (+ [action dicts]) -> trajectory dict."""
    traj = {"state": {}, "vision": {}}
    for arm in ARMS:
        traj["state"][f"{arm}_ee_pose"] = np.stack([np.asarray(o["state"][f"{arm}_ee_pose"], dtype=np.float64) for o in observations])
        traj["state"][f"{arm}_ee_joint_state"] = np.stack(
            [np.asarray(o["state"][f"{arm}_ee_joint_state"], dtype=np.float64).reshape(1) for o in observations])
    for cam in CAMERAS:
        traj["vision"][cam] = np.stack([np.asarray(o["vision"][cam]["color"], dtype=np.uint8) for o in observations])
    if actions is not None:
        assert len(actions) == len(observations), "one action per observation"
        traj["action"] = {}
        for arm in ARMS:
            traj["action"][f"{arm}_ee_pose"] = np.stack([np.asarray(a[f"{arm}_ee_pose"], dtype=np.float64) for a in actions])
            traj["action"][f"{arm}_ee_joint_state"] = np.stack(
                [np.asarray(a[f"{arm}_ee_joint_state"], dtype=np.float64).reshape(1) for a in actions])
    return traj


def demonstration_to_trajectory(demo):
    """A demonstration in any accepted form -> {"state": {...}, "action": {...}, "vision": {cam: (T, H, W, 3) RGB}}.

    Accepted: an HDF5 trajectory path; an XPolicyLab trajectory dict (`state`, `vision`, optional `action`, plural or
    singular key names, encoded or decoded images); or {"observations": [obs, ...], "actions": [action, ...]}.
    Without actions, the action at t is the state at t + 1 (the last step repeats), which is how RoboTwin records them.
    """
    if isinstance(demo, (str, os.PathLike)):
        traj = _load_hdf5(os.fspath(demo))
    elif isinstance(demo, dict) and "observations" in demo:
        traj = _stack_obs_list(demo["observations"], demo.get("actions"))
    elif isinstance(demo, dict) and "state" in demo:
        traj = {"state": {}, "vision": {}}
        for arm in ARMS:
            traj["state"][f"{arm}_ee_pose"] = np.asarray(_get(demo["state"], f"{arm}_ee_poses", f"{arm}_ee_pose"), dtype=np.float64)
            traj["state"][f"{arm}_ee_joint_state"] = np.asarray(
                _get(demo["state"], f"{arm}_ee_joint_states", f"{arm}_ee_joint_state"), dtype=np.float64).reshape(-1, 1)
        if "action" in demo:
            traj["action"] = {}
            for arm in ARMS:
                traj["action"][f"{arm}_ee_pose"] = np.asarray(_get(demo["action"], f"{arm}_ee_poses", f"{arm}_ee_pose"), dtype=np.float64)
                traj["action"][f"{arm}_ee_joint_state"] = np.asarray(
                    _get(demo["action"], f"{arm}_ee_joint_states", f"{arm}_ee_joint_state"), dtype=np.float64).reshape(-1, 1)
        for cam in CAMERAS:
            cam_data = demo["vision"][cam]
            traj["vision"][cam] = _decode_frames(cam_data["colors"] if "colors" in cam_data else cam_data["color"])
    else:
        raise TypeError(f"unsupported demonstration of type {type(demo).__name__}")

    length = len(traj["state"]["left_ee_pose"])
    if "action" not in traj:
        idx = np.minimum(np.arange(length) + 1, length - 1)
        traj["action"] = {k: v[idx] for k, v in traj["state"].items()}
    for group in ("state", "action"):
        for k, v in traj[group].items():
            assert len(v) == length, f"{group}/{k} has {len(v)} steps, expected {length}"
    for cam, frames in traj["vision"].items():
        assert len(frames) == length, f"vision/{cam} has {len(frames)} frames, expected {length}"
    return traj


def trajectory_lowdim(traj):
    """Trajectory -> replay-buffer low-dim arrays (float32): per-arm absolute pose/gripper and the 14-d action."""
    out, action = {}, []
    for arm in ARMS:
        prefix = ROBOT_PREFIXES[arm]
        pose = traj["state"][f"{arm}_ee_pose"]
        out[f"{prefix}_eef_pos"] = pose[:, :3].astype(np.float32)
        out[f"{prefix}_eef_rot_axis_angle"] = quat_wxyz_to_rotvec(pose[:, 3:7]).astype(np.float32)
        out[f"{prefix}_gripper_width"] = traj["state"][f"{arm}_ee_joint_state"].reshape(-1, 1).astype(np.float32)
        act = traj["action"][f"{arm}_ee_pose"]
        action += [act[:, :3], quat_wxyz_to_rotvec(act[:, 3:7]), traj["action"][f"{arm}_ee_joint_state"].reshape(-1, 1)]
    out["action"] = np.concatenate(action, axis=-1).astype(np.float32)
    assert out["action"].shape[1] == RAW_ACTION_DIM
    return out


def trajectory_images(traj):
    """Trajectory -> replay-buffer images {camera_*_rgb: (T, 224, 224, 3) uint8 RGB}."""
    return {key: np.stack([resize_image(f) for f in traj["vision"][cam]]) for cam, key in CAMERAS.items()}


# ---------------------------------------------------------------------------------------------- deployment
def xpolicylab_obs_to_bpp_frame(obs):
    """One live XPolicyLab observation -> one policy frame, exactly what `UmiTaskDataset` yields for that timestep
    with pose_repr abs and low-dim horizon 1: images (224, 224, 3) uint8, low-dim float32."""
    frame = {key: resize_image(obs["vision"][cam]["color"]) for cam, key in CAMERAS.items()}
    mats = {}
    for arm in ARMS:
        prefix = ROBOT_PREFIXES[arm]
        mats[arm] = ee_pose_to_mat(obs["state"][f"{arm}_ee_pose"])
        pose10d = mat_to_pose10d(mats[arm])
        frame[f"{prefix}_eef_pos"] = pose10d[:3].astype(np.float32)
        frame[f"{prefix}_eef_rot_axis_angle"] = pose10d[3:].astype(np.float32)
        frame[f"{prefix}_gripper_width"] = np.asarray(obs["state"][f"{arm}_ee_joint_state"], dtype=np.float32).reshape(1)
    for arm, other in (("left", "right"), ("right", "left")):
        rel = mat_to_pose10d(np.linalg.inv(mats[other]) @ mats[arm])  # convert_pose_mat_rep(..., 'relative')
        frame[f"gripper_{arm}_eef_pos_wrt_gripper_{other}"] = rel[:3].astype(np.float32)
        frame[f"gripper_{arm}_eef_rot_axis_angle_wrt_gripper_{other}"] = rel[3:].astype(np.float32)
    return frame


def bpp_action_to_xpolicylab(action):
    """(20,) predicted [pos 3, rot6d 6, gripper 1] x (left, right), absolute world frame -> XPolicyLab ee action dict."""
    action = np.asarray(action, dtype=np.float64)
    assert action.shape == (ACTION_DIM,), f"expected a ({ACTION_DIM},) action, got {action.shape}"
    out = {}
    for i, arm in enumerate(ARMS):
        a = action[10 * i: 10 * (i + 1)]
        mat = pose10d_to_mat(a[:9])
        out[f"{arm}_ee_pose"] = np.concatenate([mat[:3, 3], mat_to_quat_wxyz(mat[:3, :3])]).astype(np.float32)
        out[f"{arm}_ee_joint_state"] = np.asarray([np.clip(a[9], 0.0, 1.0)], dtype=np.float32)
    return out
