"""`BazarBlotAEC`: the primary environment API (spec §5.1) — a PettingZoo `AECEnv` with agents
`"player_0".."player_3"`, one seat each.

Follows the standard PettingZoo classic-game idiom (see e.g. `pettingzoo.classic.tictactoe`) for
everything the base class expects: `agents` shrinks via `_was_dead_step` as agents terminate,
`rewards`/`_cumulative_rewards` are maintained via `_clear_rewards`/`_accumulate_rewards`, and
`observation_space`/`action_space` return the same cached object every call. The one real
departure from that idiom: turn order here is NOT a fixed round-robin (`agent_selector`), because
who acts next depends on auction/trick outcomes, so `agent_selection` is set directly from
`TrackedDeal`'s `to_act` after every step instead.

**Redeals are invisible to the agent** (a documented Known Trap in the roadmap: "4-pass redeals
farmed as a safe action — aborts carry no reward and are resampled by the driver"). A 4-pass
abort inside `step()` transparently deals a new hand with the same dealer and continues within
the SAME `step()` call — no extra termination, no extra reward, no agent ever sees an aborted
deal as a distinct decision point.

**Action masking, not the spec's literal `{"obs": ..., "action_mask": ...}` key names.** The env
spec's §5.1 example uses `"obs"`; this implementation uses `"observation"` instead, matching
PettingZoo's own actual convention (`pettingzoo.classic.tictactoe` and every other bundled
action-masked env) — `pettingzoo.test.api_test`'s own dict-unwrapping logic specifically looks
for a key named `"observation"`, so matching the spec's literal wording instead would silently
skip real test coverage. Treated the same as the module's other spec-vs-reality reconciliations
(`env/actions.py`'s `ACTION_DIM`, `env/obs.py`'s `melds_public` shape): documented, not silent.
"""

from __future__ import annotations

import math
import random
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo import AECEnv

from bazarblot.core.cards import TEAM_OF
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.dealing import deal_seed as _shared_deal_seed
from bazarblot.core.match import Match
from bazarblot.core.rules import RuleConfig, load_default
from bazarblot.env.actions import ActionSpace, build_action_space, legal_mask
from bazarblot.env.infoset import InfoSet, info_set
from bazarblot.env.obs import BLOCK_SHAPES, encode, flatten
from bazarblot.env.tracked_deal import TrackedDeal

_FLAT_OBS_DIM = sum(int(np.prod(s)) for s in BLOCK_SHAPES.values())
_REWARD_SCALE = 24.0  # spec §6: tanh(delta/24), constant fixed from an empirical median margin


def _seat_of(agent: str) -> int:
    return int(agent.split("_")[1])


def _agent_of(seat: int) -> str:
    return f"player_{seat}"


class BazarBlotAEC(AECEnv):
    """`episode_unit`: `"deal"` (default, one deal per episode) or `"match"` (episode runs until
    `Match.finished`, per-deal rewards fire at every deal within it) — both spec-required first
    class modes (§5.1), independent of the `RuleConfig.match.episode_unit` default so a single
    env instance can be evaluated either way without re-loading the rules preset.
    """

    metadata: ClassVar[dict[str, object]] = {"name": "bazar_blot_v0", "is_parallelizable": False}

    def __init__(self, rules: RuleConfig | None = None, episode_unit: str = "deal") -> None:
        super().__init__()
        if episode_unit not in ("deal", "match"):
            raise ValueError(f"episode_unit must be 'deal' or 'match', got {episode_unit!r}")
        self.rules = rules if rules is not None else load_default()
        self.episode_unit = episode_unit
        self.action_space_obj: ActionSpace = build_action_space(self.rules)

        self.possible_agents = [_agent_of(s) for s in range(4)]
        self.agents = self.possible_agents[:]

        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(_FLAT_OBS_DIM,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(self.action_space_obj.action_dim,), dtype=np.int8
                ),
            }
        )
        act_space: spaces.Discrete[np.int64] = spaces.Discrete(self.action_space_obj.action_dim)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos: dict[str, dict[str, Any]] = {a: {} for a in self.agents}

        self._seed = 0
        self._redeal_attempt = 0
        self.match = Match(rules=self.rules)
        self.tracked = self._new_tracked_deal()
        self.agent_selection = _agent_of(self.tracked.deal.to_act)

    # ---------------------------------------------------------------- spaces

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    # ---------------------------------------------------------------- dealing

    def _deal_seed(self) -> int:
        return _shared_deal_seed(self._seed, self.match.deal_number, self._redeal_attempt)

    def _new_tracked_deal(self) -> TrackedDeal:
        rng = random.Random(self._deal_seed())
        hands = deal_hands(rng, self.rules)
        deal = Deal(
            self.rules, dealer=self.match.dealer, hands=hands, deal_id=self.match.deal_number
        )
        return TrackedDeal(deal)

    # ---------------------------------------------------------------- gym/pettingzoo API

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options
        self._seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
        self._redeal_attempt = 0
        self.match = Match(rules=self.rules)
        self.tracked = self._new_tracked_deal()

        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = _agent_of(self.tracked.deal.to_act)

    def _info_for(self, seat: int) -> InfoSet:
        return info_set(
            self.tracked,
            seat=seat,
            match_score=(self.match.scores[0], self.match.scores[1]),
            deal_number=self.match.deal_number,
        )

    def observe(self, agent: str) -> dict[str, npt.NDArray[Any]]:
        seat = _seat_of(agent)
        info = self._info_for(seat)
        flat = flatten(encode(info))
        acting_now = agent == self.agent_selection and self.tracked.deal.phase in (
            Phase.AUCTION,
            Phase.PLAY,
        )
        if acting_now:
            mask = legal_mask(info, self.action_space_obj).astype(np.int8)
        else:
            mask = np.zeros(self.action_space_obj.action_dim, dtype=np.int8)
        return {"observation": flat, "action_mask": mask}

    def step(self, action: int | None) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        self._clear_rewards()
        acting_agent = self.agent_selection
        seat = _seat_of(acting_agent)
        idx = int(action)  # type: ignore[arg-type]
        core_action = self.action_space_obj.decode(idx)

        legal = legal_mask(self._info_for(seat), self.action_space_obj)
        if not legal[idx]:
            raise ValueError(
                f"illegal action {idx} ({core_action!r}) for {acting_agent} at seat {seat}"
            )

        self.tracked.step(core_action)

        if self.tracked.deal.phase == Phase.ABORTED:
            self._redeal_attempt += 1
            self.tracked = self._new_tracked_deal()
            self.agent_selection = _agent_of(self.tracked.deal.to_act)
        elif self.tracked.deal.phase == Phase.TERMINAL:
            result = self.tracked.deal.result
            assert result is not None
            self.match.apply_deal_result(result)
            self._redeal_attempt = 0
            team_scores = result.team_scores
            margin_team0 = team_scores[0] - team_scores[1]
            reward_team0 = math.tanh(margin_team0 / _REWARD_SCALE)
            for s in range(4):
                self.rewards[_agent_of(s)] = reward_team0 if TEAM_OF[s] == 0 else -reward_team0

            episode_done = self.episode_unit == "deal" or self.match.finished
            if episode_done:
                self.terminations = dict.fromkeys(self.agents, True)
                self.agent_selection = acting_agent
            else:
                self.tracked = self._new_tracked_deal()
                self.agent_selection = _agent_of(self.tracked.deal.to_act)
        else:
            self.agent_selection = _agent_of(self.tracked.deal.to_act)

        # `_cumulative_rewards[agent]` means "reward accumulated since AGENT's own last turn",
        # not "total for the episode" — reset the ACTING agent's counter here (matching every
        # standard PettingZoo AECEnv), or a multi-reward-event episode (any deal in "match" mode
        # past the first) desyncs it from what `last()` promises. A single-terminal-reward
        # episode (episode_unit="deal", or any classic two-player game like tictactoe) can't
        # actually exercise this bug — there's only ever one reward event, right at the natural
        # end — which is why it's easy to omit and still pass a shallow smoke test.
        self._cumulative_rewards[acting_agent] = 0
        self._accumulate_rewards()

    def close(self) -> None:
        pass

    def render(self) -> None:
        pass
