"""Joint bid search: try my bid together with partner's reply (after Joint Policy Search, Tian,
Gong and Jiang, NeurIPS 2020).

Self-play gets stuck where no player gains by changing alone: partner never raises my suit,
because I have never learned what a raise means, so a raise really does lose points, so partner
never raises. (Measured: in the "I open 8 hearts holding the jack, partner holds the 9" case the
trained agent raised 0 times in 1,500 deals, and a forced raise lost about 2 points.) Only a JOINT
change, my bid together with partner's answer, can show that a convention pays.

On a small share of auction decisions in pure self-play deals, the deciding seat searches:

- my candidates: pass, contra/recontra when legal, and every contract type at the standing level
  plus each of `deltas` (not capot);
- for each, the left opponent answers with the current greedy policy; then partner's candidates
  (the same kinds of action) at the resulting position;
- each pair is played out to the end of the auction with the greedy policy, and the contract is
  scored by playing the deal out with the greedy agent on the real hands (0 if it is passed out).
  This uses all four hands, as the paper's double-dummy scoring does; it is training-time only and
  the network still decides from its own information. Scoring by the centralized critic instead
  is an option, but picking the best of ~100 critic estimates mostly picks the critic's error.

If the best pair beats the policy's own pair (its greedy bid and partner's greedy reply) by more
than `margin`, I play the best bid, and partner, at its next decision in this auction, searches
its best reply to the actual position. Both decisions are recorded with the network's own
probabilities (`decide(..., force_env_action=...)`, marked `forced`), so the ordinary PPO update
then reinforces the pair together. Nothing here encodes a convention: the search only compares
contracts by their value under the current agent's own card play.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from bazarblot.agents.nn.network import CentralCritic, PolicyValueNet
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import CONTRA_IDX, PASS_IDX, RECONTRA_IDX, ActionSpace, legal_mask
from bazarblot.env.canonical import (
    canonical_belief_target,
    canonical_permutation,
    permute_infoset,
)
from bazarblot.env.infoset import info_set
from bazarblot.env.obs import encode, flatten
from bazarblot.env.tracked_deal import TrackedDeal
from bazarblot.learn.belief import belief_target, privileged_features


@dataclass(frozen=True, slots=True)
class JointSearchConfig:
    prob: float = 0.05
    """Share of learner auction decisions (pure self-play deals only) that run the search."""
    deltas: tuple[int, ...] = (1, 2)
    """Bid levels tried above the standing bid."""
    margin: float = 0.02
    """Deviate only when the best pair beats the policy's own pair by more than this, in reward
    units (about 2 points of margin at reward scale 96)."""
    evaluate: str = "playout"
    """How a contract is scored: "playout" plays the deal out with the current greedy agent on the
    real hands (the exact result of this deal under the agent's own play; each distinct contract is
    played once), or "critic" asks the centralized critic (fast, but picking the best of ~100
    critic estimates mostly picks the critic's largest error)."""
    reward_scale: float = 96.0


@dataclass(slots=True)
class SearchStats:
    searches: int = 0
    deviations: int = 0
    replies: int = 0
    reply_changes: int = 0


class JointSearch:
    def __init__(
        self,
        rules: RuleConfig,
        space: ActionSpace,
        net: PolicyValueNet,
        critic: CentralCritic | None,
        config: JointSearchConfig,
    ) -> None:
        self.rules = rules
        self.space = space
        self.agent = NNAgent(net, sample=False)
        self.critic = critic
        self.config = config
        self.stats = SearchStats()

    # ------------------------------------------------------------------ public

    def search(self, tracked: TrackedDeal) -> int | None:
        """At the deciding seat's auction turn: the bid to play instead of the policy's, or None."""
        self.stats.searches += 1
        me = tracked.deal.to_act
        partner = (me + 2) % 4
        greedy_me = self._greedy(tracked)
        leaves: list[TrackedDeal | None] = []
        index: list[tuple[int, int | None, bool]] = []  # (my action, partner action, policy pair)
        for a1 in self._candidates(tracked, include=greedy_me):
            t1 = self._replay(tracked, [a1])
            self._advance_until(t1, partner)
            if t1.deal.phase != Phase.AUCTION or t1.deal.to_act != partner:
                leaves.append(self._finish(t1))
                index.append((a1, None, a1 == greedy_me))
                continue
            greedy_partner = self._greedy(t1)
            for a2 in self._candidates(t1, include=greedy_partner):
                leaves.append(self._finish(self._replay(t1, [a2])))
                index.append((a1, a2, a1 == greedy_me and a2 == greedy_partner))
        values = self._evaluate(leaves, me)
        policy_value = next(v for v, (_, _, pol) in zip(values, index, strict=True) if pol)
        best = int(np.argmax(values))
        if values[best] > policy_value + self.config.margin and index[best][0] != greedy_me:
            self.stats.deviations += 1
            return index[best][0]
        return None

    def best_reply(self, tracked: TrackedDeal) -> int | None:
        """Partner's turn right after a searched deviation: its best answer, or None to play the
        policy's own action."""
        self.stats.replies += 1
        me = tracked.deal.to_act
        greedy = self._greedy(tracked)
        actions = self._candidates(tracked, include=greedy)
        leaves = [self._finish(self._replay(tracked, [a])) for a in actions]
        values = self._evaluate(leaves, me)
        policy_value = values[actions.index(greedy)]
        best = int(np.argmax(values))
        if values[best] > policy_value + self.config.margin and actions[best] != greedy:
            self.stats.reply_changes += 1
            return actions[best]
        return None

    # ------------------------------------------------------------------ helpers

    def _candidates(self, tracked: TrackedDeal, include: int) -> list[int]:
        info = info_set(tracked, tracked.deal.to_act, (0, 0), 0)
        mask = legal_mask(info, self.space)
        out = [i for i in (PASS_IDX, CONTRA_IDX, RECONTRA_IDX) if mask[i]]
        standing = info.auction_state.standing_bid
        base = standing.level if standing is not None else self.rules.auction.min_bid - 1
        capot = bool(standing is not None and standing.capot)  # once capot is bid, bids must be
        for d in self.config.deltas:
            level = base + d
            if level > self.rules.auction.max_bid:
                continue
            for t in self.space.contract_types:
                idx = self.space.bid_index(level, t, capot)
                if mask[idx]:
                    out.append(idx)
        if include not in out:
            out.append(include)
        return out

    def _greedy(self, tracked: TrackedDeal) -> int:
        info = info_set(tracked, tracked.deal.to_act, (0, 0), 0)
        return self.agent.act(info, self.space, legal_mask(info, self.space))

    def _replay(self, tracked: TrackedDeal, extra: list[int]) -> TrackedDeal:
        """A fresh copy of the deal's auction so far, plus `extra` actions."""
        d = tracked.deal
        deal = Deal(self.rules, dealer=d.dealer, hands=d.original_hands, deal_id=d.deal_id)
        copy = TrackedDeal(deal)
        for e in tracked.auction_log:
            copy.step(e.action)
        for a in extra:
            copy.step(self.space.decode(a))
        return copy

    def _capped(self, tracked: TrackedDeal) -> bool:
        return len(tracked.auction_log) >= self.rules.auction.max_auction_steps - 1

    def _advance_until(self, tracked: TrackedDeal, seat: int) -> None:
        while (
            tracked.deal.phase == Phase.AUCTION
            and tracked.deal.to_act != seat
            and not self._capped(tracked)
        ):
            tracked.step(self.space.decode(self._greedy(tracked)))

    def _finish(self, tracked: TrackedDeal) -> TrackedDeal | None:
        """Play the rest of the auction greedily; None when it ends without a contract."""
        while tracked.deal.phase == Phase.AUCTION:
            if self._capped(tracked):
                return None
            tracked.step(self.space.decode(self._greedy(tracked)))
        return tracked if tracked.deal.phase == Phase.PLAY else None

    def _evaluate(self, leaves: list[TrackedDeal | None], seat: int) -> list[float]:
        """Each leaf's value for `seat` in reward units; 0 if the deal is passed out."""
        if self.config.evaluate == "critic":
            return self._evaluate_critic(leaves, seat)
        return self._evaluate_playout(leaves, seat)

    def _evaluate_playout(self, leaves: list[TrackedDeal | None], seat: int) -> list[float]:
        cache: dict[tuple[object, ...], float] = {}
        values = []
        for t in leaves:
            if t is None:
                values.append(0.0)
                continue
            c = t.deal.contract
            assert c is not None
            key = (c.level, c.contract_type, c.capot, c.declarer_seat, c.doubling)
            if key not in cache:
                cache[key] = self._play_out(t, seat)
            values.append(cache[key])
        return values

    def _play_out(self, tracked: TrackedDeal, seat: int) -> float:
        deal = tracked.deal
        while deal.phase == Phase.PLAY:
            tracked.step(self.space.decode(self._greedy(tracked)))
        assert deal.result is not None
        s0, s1 = deal.result.team_scores
        margin = (s0 - s1) if seat % 2 == 0 else (s1 - s0)
        return math.tanh(margin / self.config.reward_scale)

    def _evaluate_critic(self, leaves: list[TrackedDeal | None], seat: int) -> list[float]:
        """The critic's value of each contract for `seat` at the start of card play."""
        rows, priv, where = [], [], []
        for i, t in enumerate(leaves):
            if t is None:
                continue
            info = info_set(t, seat, (0, 0), 0)
            perm = canonical_permutation(info)
            rows.append(flatten(encode(permute_infoset(info, perm))))
            target = canonical_belief_target(belief_target(t.deal.hands, seat), perm)
            priv.append(privileged_features(target))
            where.append(i)
        values = [0.0] * len(leaves)  # a passed-out deal scores 0
        if rows:
            assert self.critic is not None, "critic evaluation needs a critic"
            with torch.no_grad():
                v = self.critic(torch.from_numpy(np.stack(rows)), torch.from_numpy(np.stack(priv)))
            for i, x in zip(where, v.tolist(), strict=True):
                values[i] = float(x)
        return values
