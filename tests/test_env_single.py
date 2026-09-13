"""M3: `SingleAgentEnv`, the Gym-API convenience wrapper (spec §5.2)."""

from __future__ import annotations

import numpy as np

from bazarblot.env.aec import BazarBlotAEC
from bazarblot.env.single import SingleAgentEnv, random_legal_policy


def test_full_episodes_deal_mode() -> None:
    env = SingleAgentEnv(BazarBlotAEC(episode_unit="deal"))
    for ep in range(60):
        obs, _info = env.reset(seed=ep)
        assert env.observation_space.contains(obs)
        steps = 0
        while True:
            action = random_legal_policy(obs)
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, _info = env.step(action)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            steps += 1
            if terminated or truncated:
                break
            assert steps < 500, "runaway episode"


def test_full_episodes_match_mode() -> None:
    env = SingleAgentEnv(BazarBlotAEC(episode_unit="match"))
    for ep in range(30):
        obs, _info = env.reset(seed=ep)
        steps = 0
        while True:
            action = random_legal_policy(obs)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
            assert steps < 3000, "runaway episode"


def test_seat_rotates_across_episodes() -> None:
    env = SingleAgentEnv(BazarBlotAEC(episode_unit="deal"), rotate_seat=True)
    seats_seen = set()
    for ep in range(8):
        env.reset(seed=ep)
        seats_seen.add(env.learner_seat)
    assert seats_seen == {0, 1, 2, 3}


def test_seat_fixed_when_rotation_disabled() -> None:
    env = SingleAgentEnv(BazarBlotAEC(episode_unit="deal"), rotate_seat=False)
    for ep in range(8):
        env.reset(seed=ep)
        assert env.learner_seat == 0


def test_observation_space_matches_actual_observations() -> None:
    env = SingleAgentEnv(BazarBlotAEC(episode_unit="deal"))
    obs, _ = env.reset(seed=0)
    assert obs["observation"].dtype == np.float32
    assert obs["action_mask"].dtype == np.int8
    assert env.observation_space["observation"].shape == obs["observation"].shape
    assert env.observation_space["action_mask"].shape == obs["action_mask"].shape
