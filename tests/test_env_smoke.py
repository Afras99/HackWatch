"""
Smoke tests: reset → step → state roundtrip.
Verifies reward in [0,1] and matches hackathon validator expectations.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from hackwatch.models import MonitorAction, MonitorObservation, HackWatchState
from server.environment import HackWatchEnvironment


@pytest.fixture()
def env():
    return HackWatchEnvironment(seed=42)


def test_reset_returns_observation(env):
    obs = env.reset(seed=0)
    assert isinstance(obs, MonitorObservation)
    assert isinstance(obs.task_prompt, str) and obs.task_prompt
    assert isinstance(obs.step_idx, int)
    assert obs.episode_done is False


def test_step_returns_obs_and_reward(env):
    env.reset(seed=1)
    action = MonitorAction(verdict="allow", confidence=0.1, reasoning="test")
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, MonitorObservation)
    # reward is None until episode ends; done may be True if single-step episode
    if done:
        assert reward is not None
        assert 0.0 <= reward <= 1.0


def test_block_ends_episode(env):
    env.reset(seed=2)
    action = MonitorAction(verdict="block", confidence=0.9, reasoning="suspicious")
    obs, reward, done, info = env.step(action)
    assert done is True
    assert reward is not None
    assert 0.0 <= reward <= 1.0


def test_state_after_reset(env):
    env.reset(seed=3)
    state = env.state
    assert isinstance(state, HackWatchState)
    assert state.difficulty in ("train", "heldout_novel")
    assert state.task_id != ""
    assert state.episode_id != ""


def test_planted_label_not_in_public_state(env):
    """planted_label must NOT appear in the public state dict (no leakage)."""
    env.reset(seed=4)
    public = env.state.to_dict()
    assert "planted_label" not in public, (
        "planted_label leaked into public state — monitor could cheat!"
    )


def test_multiple_steps(env):
    env.reset(seed=5)
    obs, reward, done, info = env.step(MonitorAction(verdict="allow", confidence=0.2))
    if not done:
        obs2, reward2, done2, info2 = env.step(MonitorAction(verdict="flag", confidence=0.7))
        assert obs2.step_idx >= obs.step_idx


def test_reward_components_present(env):
    env.reset(seed=6)
    _, reward, done, info = env.step(MonitorAction(verdict="block", confidence=0.8))
    if done:
        assert "detection" in info
        assert "calibration_penalty" in info
        assert "latency_bonus" in info
        assert "planted_label" in info   # OK in info — not in MonitorObservation


def test_obs_serialization_roundtrip(env):
    obs = env.reset(seed=7)
    d = obs.to_dict()
    obs2 = MonitorObservation.from_dict(d)
    assert obs2.task_prompt == obs.task_prompt
    assert obs2.step_idx == obs.step_idx
    assert obs2.episode_done == obs.episode_done
