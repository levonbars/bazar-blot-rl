"""How much is knowing partner's hand worth? (the ceiling for partnership signaling)

A network trained with the oracle input (`agents/nn/network.py::ORACLE_DIM`, partner's hand shown
in training) plays as one team, seeing partner's hand; a baseline network plays the other team
from its ordinary information. Every deal is played twice with the teams swapped and the same
cards, so deal luck cancels. Both teams play greedily. A deal that everyone passes counts 0.

The result bounds what perfect bidding communication could add against that baseline: bids can at
best tell partner what the oracle shows outright. With `see=False` the oracle team plays blind
(oracle input empty), which separates "sees partner's hand" from "is a different network".
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bazarblot.agents.nn.network import PolicyValueNet
from bazarblot.agents.nn.ppo_agent import decide
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal
from bazarblot.eval.metrics import ConfidenceInterval, bootstrap_ci


@dataclass(frozen=True, slots=True)
class OracleReport:
    margin: ConfidenceInterval
    """Mean score margin per deal of the oracle team, over pairs of deals."""
    n_pairs: int
    declare_rate: float
    """Share of completed deals where the oracle team won the auction."""

    def summary(self) -> str:
        m = self.margin
        return (
            f"margin={m.mean:+.2f} CI=({m.lo:+.2f}, {m.hi:+.2f}) n_pairs={self.n_pairs} "
            f"oracle_team_declares={self.declare_rate:.2f}"
        )


def _play(
    rules: RuleConfig,
    space: ActionSpace,
    hands: tuple[frozenset[int], ...],
    dealer: int,
    oracle_net: PolicyValueNet,
    base_net: PolicyValueNet,
    oracle_team: int,
    see: bool,
) -> tuple[float, bool | None]:
    deal = Deal(rules, dealer=dealer, hands=hands, deal_id=0)
    tracked = TrackedDeal(deal)
    while deal.phase in (Phase.AUCTION, Phase.PLAY):
        if (
            deal.phase == Phase.AUCTION
            and len(tracked.auction_log) >= rules.auction.max_auction_steps
        ):
            return 0.0, None
        seat = deal.to_act
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        mask = legal_mask(info, space)
        if seat % 2 == oracle_team:
            cards = deal.hands[(seat + 2) % 4] if see else frozenset()
            d = decide(oracle_net, info, space, mask, sample=False, oracle_cards=cards)
        else:
            d = decide(base_net, info, space, mask, sample=False)
        tracked.step(space.decode(d.env_action_idx))
    if deal.result is None:
        return 0.0, None
    s = deal.result.team_scores
    margin = s[oracle_team] - s[1 - oracle_team]
    return float(margin), deal.result.attackers_team == oracle_team


def oracle_match(
    rules: RuleConfig,
    space: ActionSpace,
    oracle_net: PolicyValueNet,
    base_net: PolicyValueNet,
    n_pairs: int = 300,
    seed: int = 4242,
    see: bool = True,
) -> OracleReport:
    rng = random.Random(seed)
    margins: list[float] = []
    declared: list[bool] = []
    oracle_net.eval()
    base_net.eval()
    for i in range(n_pairs):
        hands = tuple(deal_hands(rng, rules))
        pair = []
        for team in (0, 1):
            m, d = _play(rules, space, hands, i % 4, oracle_net, base_net, team, see)
            pair.append(m)
            if d is not None:
                declared.append(d)
        margins.append(sum(pair) / 2)
    return OracleReport(
        margin=bootstrap_ci(margins, seed=seed),
        n_pairs=n_pairs,
        declare_rate=sum(declared) / max(1, len(declared)),
    )
