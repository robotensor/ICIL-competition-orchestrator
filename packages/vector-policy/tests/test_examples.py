import importlib

import numpy as np
import pytest

from vector_policy import ACTION_TYPES, Policy, manifest

NAMES = ("replay_policy", "zero_policy")


def build(examples, name, monkeypatch):
    """The example's policy, built the way the server builds it: from its manifest's root."""
    loaded = manifest.load(examples / name / "policy.yaml")
    monkeypatch.syspath_prepend(str(loaded.root))
    cls = getattr(importlib.import_module(loaded.module), loaded.attribute)
    return cls(**loaded.kwargs)


@pytest.mark.parametrize("name", NAMES)
def test_each_example_is_a_complete_repository_with_a_valid_manifest(examples, name):
    loaded = manifest.load(examples / name / "policy.yaml")
    assert loaded.requirements_path is not None and loaded.requirements_path.is_file()
    assert (examples / name / "README.md").is_file()
    assert (loaded.root / loaded.module.replace(".", "/")).with_suffix(".py").is_file()


@pytest.mark.parametrize("name", NAMES)
def test_each_example_satisfies_the_policy_protocol(examples, name, monkeypatch):
    policy = build(examples, name, monkeypatch)
    assert isinstance(policy, Policy)
    assert policy.action_type in ACTION_TYPES


def test_replay_returns_the_kth_demonstration_action_after_reset_then_the_last(
    examples, demonstration, observe, monkeypatch
):
    arrays, info = demonstration
    policy = build(examples, "replay_policy", monkeypatch)
    assert policy.action_type == "qpos"
    policy.set_demonstration(arrays, info)
    policy.reset(0)
    actions = arrays["actions"]
    for k in range(len(actions) + 3):
        got = policy.act(observe(arrays, min(k, len(actions))))
        np.testing.assert_array_equal(got["action"], actions[min(k, len(actions) - 1)])
    policy.reset(1)
    np.testing.assert_array_equal(policy.act(observe(arrays))["action"], actions[0])


def test_replay_refuses_a_demonstration_without_actions_and_acting_before_one(
    examples, monkeypatch
):
    policy = build(examples, "replay_policy", monkeypatch)
    with pytest.raises(RuntimeError, match="before set_demonstration"):
        policy.act({})
    with pytest.raises(ValueError, match="no 'actions'"):
        policy.set_demonstration({"qpos": np.zeros((3, 16))}, {})
    with pytest.raises(ValueError, match="non-empty"):
        policy.set_demonstration({"actions": np.zeros((0, 16))}, {})


def test_zero_returns_zeros_shaped_like_an_action_row(
    examples, demonstration, observe, monkeypatch
):
    arrays, info = demonstration
    policy = build(examples, "zero_policy", monkeypatch)
    assert policy.action_type == "qpos"
    policy.set_demonstration(arrays, info)
    policy.reset(0)
    for t in range(3):
        action = policy.act(observe(arrays, t))["action"]
        assert action.shape == arrays["actions"].shape[1:] == (16,)
        assert action.dtype == arrays["actions"].dtype
        assert not action.any()
