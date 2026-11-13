"""Fixed bidding situations that check partnership bidding directly (evaluation only).

`heart_support`: the project owner's example. I open 8 hearts holding the jack and two more
hearts, the next player passes, and partner decides. With the 9 and one more heart ("fit") partner
should show support (raise); with no heart at all ("no fit") it should not. Hands are random apart
from those constraints. The trained v4 agent (run A) raised 0 times in 1,500 fit deals.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bazarblot.agents.base import Agent
from bazarblot.core.auction import BidAction, PassAction
from bazarblot.core.cards import RANKS, card_id
from bazarblot.core.deal import Deal
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal


@dataclass(frozen=True, slots=True)
class SupportReport:
    fit: dict[str, float]  # raise / pass / other suit, partner holding the 9 + one heart
    no_fit: dict[str, float]  # same, partner holding no heart
    n: int

    def summary(self) -> str:
        f = "  ".join(f"{k} {v:.2f}" for k, v in self.fit.items())
        nf = "  ".join(f"{k} {v:.2f}" for k, v in self.no_fit.items())
        return f"fit: {f}  |  no fit: {nf}  (n={self.n} each)"


def _hands(rng: random.Random, fit: bool) -> tuple[frozenset[int], ...]:
    hearts = [card_id("H", r) for r in RANKS]
    jack, nine = card_id("H", "J"), card_id("H", "9")
    small = [c for c in hearts if c not in (jack, nine)]
    rng.shuffle(small)
    opener = {jack, small[0], small[1]}
    partner: set[int] = {nine, small[2]} if fit else set()
    opp_hearts = small[3:] if fit else [nine, *small[2:]]
    rest = [c for c in range(32) if c not in hearts]
    rng.shuffle(rest)
    opener |= set(rest[:5])
    rest = rest[5:]
    k = 8 - len(partner)
    partner |= set(rest[:k])
    pool = rest[k:] + opp_hearts
    rng.shuffle(pool)
    return (frozenset(opener), frozenset(pool[:8]), frozenset(partner), frozenset(pool[8:]))


def heart_support(
    rules: RuleConfig, space: ActionSpace, agent: Agent, n_deals: int = 300, seed: int = 11
) -> SupportReport:
    rng = random.Random(seed)
    out = {}
    for fit in (True, False):
        counts = {"raise": 0, "pass": 0, "other": 0}
        for _ in range(n_deals):
            deal = Deal(rules, dealer=3, hands=_hands(rng, fit), deal_id=0)
            tracked = TrackedDeal(deal)
            tracked.step(BidAction(8, "H"))
            tracked.step(PassAction())
            info = info_set(tracked, 2, (0, 0), 0)
            action = space.decode(agent.act(info, space, legal_mask(info, space)))
            if isinstance(action, BidAction) and action.contract_type == "H":
                counts["raise"] += 1
            elif isinstance(action, PassAction):
                counts["pass"] += 1
            else:
                counts["other"] += 1
        out[fit] = {k: v / n_deals for k, v in counts.items()}
    return SupportReport(fit=out[True], no_fit=out[False], n=n_deals)
