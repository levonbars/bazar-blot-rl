"""M4: `RandomAgent` — the floor."""

from __future__ import annotations

import random

from bazarblot.agents.random_agent import RandomAgent
from bazarblot.env.aec import BazarBlotAEC


def test_random_agent_only_ever_plays_legal_actions() -> None:
    env = BazarBlotAEC(episode_unit="deal")
    agent = RandomAgent(random.Random(0))
    for ep in range(30):
        env.reset(seed=ep)
        steps = 0
        while env.agents and steps < 300:
            a = env.agent_selection
            if env.terminations[a] or env.truncations[a]:
                env.step(None)
                steps += 1
                continue
            seat = int(a.split("_")[1])
            info = env._info_for(seat)
            obs = env.observe(a)
            mask = obs["action_mask"].astype(bool)
            action = agent.act(info, env.action_space_obj, mask)
            assert mask[action]
            env.step(action)
            steps += 1


def test_random_agent_is_deterministic_given_a_seeded_rng() -> None:
    env = BazarBlotAEC(episode_unit="deal")
    agent_a = RandomAgent(random.Random(7))
    agent_b = RandomAgent(random.Random(7))
    env.reset(seed=1)
    steps = 0
    while env.agents and steps < 200:
        a = env.agent_selection
        if env.terminations[a] or env.truncations[a]:
            env.step(None)
            steps += 1
            continue
        seat = int(a.split("_")[1])
        info = env._info_for(seat)
        mask = env.observe(a)["action_mask"].astype(bool)
        chosen_a = agent_a.act(info, env.action_space_obj, mask)
        chosen_b = agent_b.act(info, env.action_space_obj, mask)
        assert chosen_a == chosen_b
        env.step(chosen_a)
        steps += 1
