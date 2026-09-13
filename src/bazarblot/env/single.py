"""`SingleAgentEnv` (spec §5.2): a standard Gym API wrapper around `BazarBlotAEC` with one
learning seat and three fixed opponents. For evaluation and quick baselines, not self-play
training (`BazarBlotAEC` is the training-facing API — see its module docstring).

Opponents are any `Callable[[dict[str, np.ndarray]], int]` taking an `observe()`-shaped dict
(`{"observation": ..., "action_mask": ...}`) and returning a legal flat action index.
`random_legal_policy` is the only one shipped here — the "club player" heuristic and PIMC agents
are M4's job (`agents/`), not this env layer.
"""

from __future__ import annotations

import random as _random
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from bazarblot.env.aec import BazarBlotAEC, _agent_of

ObsDict = dict[str, npt.NDArray[Any]]
OpponentPolicy = Callable[[ObsDict], int]


def random_legal_policy(obs: ObsDict, rng: _random.Random | None = None) -> int:
    legal = obs["action_mask"].nonzero()[0]
    chooser = rng.choice if rng is not None else _random.choice
    return int(chooser(list(legal)))


class SingleAgentEnv(gym.Env[ObsDict, int]):
    # Matches `gym.Env`'s own base-class declaration (`metadata: dict[str, Any]`, a plain
    # instance attribute, not a ClassVar) — annotating it as ClassVar here would conflict with
    # the base under mypy, even though it IS one in practice.
    metadata: dict[str, Any] = {"name": "bazar_blot_single_v0"}  # noqa: RUF012

    def __init__(
        self,
        aec_env: BazarBlotAEC | None = None,
        opponent_policy: OpponentPolicy | None = None,
        rotate_seat: bool = True,
    ) -> None:
        self.aec = aec_env if aec_env is not None else BazarBlotAEC()
        self.opponent_policy = (
            opponent_policy if opponent_policy is not None else random_legal_policy
        )
        self.rotate_seat = rotate_seat
        self._episode_count = 0
        self.learner_seat = 0

        agent0 = _agent_of(0)
        # `self.action_space`/`self.observation_space` are gym.Env's own attributes (typed
        # generically as `Space[Any]` by the base class) — kept for gym API compliance. The
        # private `_obs_space` reference below is separately typed as the concrete `spaces.Dict`
        # it actually is, needed wherever this class indexes into it by key (`["observation"]`),
        # which a bare `Space[Any]` doesn't support under mypy.
        self.action_space = self.aec.action_space(agent0)
        self.observation_space = self.aec.observation_space(agent0)
        obs_space = self.aec.observation_space(agent0)
        assert isinstance(obs_space, spaces.Dict)
        self._obs_space: spaces.Dict = obs_space

    def _learner_agent(self) -> str:
        return _agent_of(self.learner_seat)

    def _run_opponents_until_learner_or_done(self) -> float:
        """Step every non-learner agent automatically until it's the learner's turn or the
        episode has ended, returning whatever reward the learner accrued along the way.

        Reads `self.aec.rewards[learner]` (the PER-STEP reward) right after each underlying
        `aec.step()` call rather than `_cumulative_rewards` at the end — once the episode ends,
        cleaning up the other three agents via `_was_dead_step` can cascade into cleaning up the
        learner's own agent too (all four terminate simultaneously), which deletes its
        `_cumulative_rewards` entry before this method returns. Reading the per-step delta as it
        happens is immune to that ordering."""
        learner = self._learner_agent()
        reward = 0.0
        while self.aec.agents and self.aec.agent_selection != learner:
            agent = self.aec.agent_selection
            if self.aec.terminations[agent] or self.aec.truncations[agent]:
                self.aec.step(None)
            else:
                obs = self.aec.observe(agent)
                action = self.opponent_policy(obs)
                self.aec.step(action)
            reward += self.aec.rewards.get(learner, 0.0)
        return reward

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsDict, dict[str, Any]]:
        super().reset(seed=seed)
        self.learner_seat = self._episode_count % 4 if self.rotate_seat else 0
        self._episode_count += 1
        self.aec.reset(seed=seed, options=options)
        self._run_opponents_until_learner_or_done()
        obs = self.aec.observe(self._learner_agent())
        return obs, {}

    def step(self, action: int) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        agent = self._learner_agent()
        self.aec.step(action)
        # Read the learner's own move's reward immediately, before the opponent fast-forward
        # below can possibly touch anything — see `_run_opponents_until_learner_or_done`'s
        # docstring for why per-step `.rewards` is read rather than `._cumulative_rewards`.
        reward = self.aec.rewards.get(agent, 0.0)
        reward += self._run_opponents_until_learner_or_done()

        terminated = self.aec.terminations.get(agent, True)
        truncated = self.aec.truncations.get(agent, False)
        if agent in self.aec.agents:
            obs = self.aec.observe(agent)
        else:
            # Learner's own agent was already cleaned up (dead-stepped) once the episode ended —
            # return a harmless zeroed observation; the caller must not act on a done episode.
            obs_space = self._obs_space["observation"]
            mask_space = self._obs_space["action_mask"]
            assert isinstance(obs_space, spaces.Box) and isinstance(mask_space, spaces.Box)
            obs = {
                "observation": np.zeros(obs_space.shape, dtype=np.float32),
                "action_mask": np.zeros(mask_space.shape, dtype=np.int8),
            }
        return obs, reward, terminated, truncated, {}

    def render(self) -> None:
        self.aec.render()

    def close(self) -> None:
        self.aec.close()
