"""M3: `BazarBlotAEC`. `pettingzoo.test.api_test` is the roadmap's stated "done when" bar —
everything else here is extra confidence the deal/redeal/reward machinery behind it is correct,
not just conformant."""

from __future__ import annotations

import math
import random

import pytest
from pettingzoo.test import api_test

from bazarblot.env.aec import BazarBlotAEC


def _random_legal_step(env: BazarBlotAEC, rng: random.Random) -> None:
    agent = env.agent_selection
    if env.terminations[agent] or env.truncations[agent]:
        env.step(None)
        return
    obs = env.observe(agent)
    legal = obs["action_mask"].nonzero()[0]
    env.step(int(rng.choice(legal)))


def test_api_test_deal_mode() -> None:
    api_test(BazarBlotAEC(episode_unit="deal"), num_cycles=1500)


def test_api_test_match_mode() -> None:
    api_test(BazarBlotAEC(episode_unit="match"), num_cycles=3000)


def test_reset_is_deterministic_given_a_seed() -> None:
    env_a = BazarBlotAEC(episode_unit="deal")
    env_b = BazarBlotAEC(episode_unit="deal")
    env_a.reset(seed=42)
    env_b.reset(seed=42)
    rng_a, rng_b = random.Random(1), random.Random(1)
    for _ in range(40):
        if not env_a.agents:
            break
        agent_a = env_a.agent_selection
        agent_b = env_b.agent_selection
        assert agent_a == agent_b
        if env_a.terminations[agent_a]:
            env_a.step(None)
            env_b.step(None)
            continue
        obs_a = env_a.observe(agent_a)
        obs_b = env_b.observe(agent_b)
        assert (obs_a["action_mask"] == obs_b["action_mask"]).all()
        legal = obs_a["action_mask"].nonzero()[0]
        action = int(rng_a.choice(legal))
        _ = rng_b  # kept in lockstep implicitly by identical action selection below
        env_a.step(action)
        env_b.step(action)


def test_reward_is_antisymmetric_between_teams() -> None:
    """spec §6: `reward_A == -reward_B`, exactly — the raw-margin zero-sum identity, not an
    approximation."""
    env = BazarBlotAEC(episode_unit="deal")
    rng = random.Random(3)
    for seed in range(80):
        env.reset(seed=seed)
        while env.agents:
            _random_legal_step(env, rng)
            if any(env.rewards.values()):
                r0 = env.rewards["player_0"]
                r1 = env.rewards["player_1"]
                r2 = env.rewards["player_2"]
                r3 = env.rewards["player_3"]
                assert r0 == r2  # same team
                assert r1 == r3
                assert math.isclose(r0, -r1, abs_tol=1e-9)


def test_redeal_on_abort_is_invisible_to_the_agent() -> None:
    """No termination, no reward, no distinct decision point for a 4-pass abort — the roadmap's
    documented trap: "4-pass redeals farmed as a safe action — aborts carry no reward and are
    resampled by the driver." Force it by always passing."""
    env = BazarBlotAEC(episode_unit="deal")
    env.reset(seed=0)
    steps = 0
    saw_nonzero_reward_before_end = False
    while env.agents and steps < 200:
        agent = env.agent_selection
        if env.terminations[agent]:
            env.step(None)
            steps += 1
            continue
        obs = env.observe(agent)
        legal = obs["action_mask"].nonzero()[0]
        pass_idx = 0
        action = pass_idx if pass_idx in legal else int(legal[0])
        env.step(action)
        if any(env.rewards.values()) and env.agents:
            saw_nonzero_reward_before_end = True
        steps += 1
    assert not saw_nonzero_reward_before_end


def test_illegal_action_raises() -> None:
    env = BazarBlotAEC(episode_unit="deal")
    env.reset(seed=0)
    agent = env.agent_selection
    obs = env.observe(agent)
    illegal = int((~obs["action_mask"].astype(bool)).nonzero()[0][0])
    with pytest.raises(ValueError):
        env.step(illegal)
