"""The benchmark's arrays as XPolicyLab training data and live observations. numpy and OpenCV."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from vector_runtime import demo  # noqa: E402

#: RoboTwin's copy of XPolicyLab's image codec, the one its episode writer used.
ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT", "/root/robotensor/vector/RoboTwin-Vector"))


def demonstration(count: int = 5, height: int = 24, width: int = 32) -> dict:
    rng = np.random.default_rng(0)
    arrays = {
        f"frames_{camera}": rng.integers(0, 256, (count, height, width, 3), dtype=np.uint8)
        for camera in demo.CAMERAS
    }
    endpose = np.zeros((count, 16))
    endpose[:, 0] = np.linspace(-0.4, -0.3, count)
    endpose[:, 3] = 1.0  # identity quaternions
    endpose[:, 7] = np.linspace(1.0, 0.0, count)
    endpose[:, 8] = 0.4
    endpose[:, 11] = 1.0
    endpose[:, 15] = 1.0
    arrays["endpose"] = endpose
    arrays["qpos"] = np.zeros((count, 16))
    arrays["actions"] = np.zeros((count - 1, 16))
    arrays["times"] = np.arange(count) / 15.0
    arrays["frequency"] = np.asarray(15.0)
    for value in arrays.values():
        value.setflags(write=False)  # arrays arrive read-only
    return arrays


def test_a_demonstration_drops_its_last_frame_and_pairs_state_with_the_next():
    arrays = demonstration()
    traj = demo.demonstration_trajectory(arrays)
    endpose = arrays["endpose"]
    np.testing.assert_array_equal(traj["state"]["left_ee_pose"], endpose[:-1, 0:7])
    np.testing.assert_array_equal(traj["action"]["left_ee_pose"], endpose[1:, 0:7])
    np.testing.assert_array_equal(traj["state"]["left_ee_joint_state"], endpose[:-1, 7:8])
    np.testing.assert_array_equal(traj["action"]["right_ee_pose"], endpose[1:, 8:15])
    np.testing.assert_array_equal(traj["action"]["right_ee_joint_state"], endpose[1:, 15:16])
    assert sorted(traj["vision"]) == ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    frames = traj["vision"]["cam_head"]["color"]
    assert frames.shape == (4, 24, 32, 3) and frames.dtype == np.uint8


def test_demonstration_frames_go_through_jpeg():
    arrays = demonstration()
    stored = demo.demonstration_trajectory(arrays)["vision"]["cam_left_wrist"]["color"]
    original = arrays["frames_left_camera"][:-1]
    assert not np.array_equal(stored, original)  # noise does not survive JPEG
    np.testing.assert_array_equal(stored[1], demo.jpeg_roundtrip(original[1]))


@pytest.mark.skipif(
    not (ROBOTWIN / "data" / "decode_image_bit.py").exists(), reason="no RoboTwin checkout"
)
def test_the_jpeg_round_trip_is_xpolicylabs_encode_then_decode():
    spec = importlib.util.spec_from_file_location("codec", ROBOTWIN / "data/decode_image_bit.py")
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    frames = demonstration(count=3, height=240, width=320)["frames_head_camera"]
    for frame in frames:
        expected = codec.decode_image_bit(codec.encode_image_bit(frame))
        np.testing.assert_array_equal(demo.jpeg_roundtrip(frame), expected)


def test_an_observation_is_the_live_xpolicylab_observation_without_jpeg():
    arrays = demonstration()
    row = {k: v[2] for k, v in arrays.items() if k.startswith("frames_") or k == "endpose"}
    obs = demo.observation(row)
    np.testing.assert_array_equal(
        obs["vision"]["cam_right_wrist"]["color"], row["frames_right_camera"]
    )
    np.testing.assert_array_equal(obs["state"]["right_ee_pose"], arrays["endpose"][2, 8:15])
    np.testing.assert_array_equal(obs["state"]["left_ee_joint_state"], arrays["endpose"][2, 7:8])


def test_an_action_dict_is_the_benchmarks_row():
    action = {
        "left_ee_pose": np.arange(7, dtype=np.float32),
        "left_ee_joint_state": np.asarray([0.5], dtype=np.float32),
        "right_ee_pose": np.arange(7, 14, dtype=np.float32),
        "right_ee_joint_state": np.asarray([1.0], dtype=np.float32),
    }
    row = demo.action_row(action)
    assert row.dtype == np.float64 and row.shape == (16,)
    np.testing.assert_array_equal(row, [0, 1, 2, 3, 4, 5, 6, 0.5, 7, 8, 9, 10, 11, 12, 13, 1.0])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a.pop("frames_head_camera"), "frames_head_camera"),
        (
            lambda a: a.update(frames_left_camera=a["frames_left_camera"].astype(np.float32)),
            "uint8",
        ),
        (lambda a: a.update(endpose=a["endpose"][:, :14]), "endpose"),
        (lambda a: a.update(endpose=np.full((5, 16), np.nan)), "non-finite"),
        (lambda a: a.update(frames_right_camera=a["frames_right_camera"][:4]), "4 frames"),
    ],
)
def test_a_bad_demonstration_is_refused(change, message):
    arrays = dict(demonstration())
    change(arrays)
    with pytest.raises(demo.DemonstrationError, match=message):
        demo.demonstration_trajectory(arrays)


def test_one_frame_is_not_a_demonstration():
    arrays = {k: (v[:1] if k != "frequency" else v) for k, v in demonstration().items()}
    with pytest.raises(demo.DemonstrationError, match="at least 2"):
        demo.demonstration_trajectory(arrays)
