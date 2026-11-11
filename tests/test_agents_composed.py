"""`ComposedAgent` routes by phase: auction decisions to the bidder, card play to the player.
The property that matters for the two-phase experiments is that a composed agent's bids are
EXACTLY its bidder's and its card plays are EXACTLY its player's -- so a margin between two
composed agents sharing a bidder can only come from play."""

from __future__ import annotations

import random

from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal

RULES = load_default()
SPACE = build_action_space(RULES)


def test_composed_agent_bids_like_bidder_and_plays_like_player() -> None:
    bidder = HeuristicAgent(RULES)
    player = RandomAgent(random.Random(0))
    composed = ComposedAgent(bidder, player)
    reference_player = RandomAgent(random.Random(0))  # identical stream to `player`

    rng = random.Random(3)
    n_bid = n_play = 0
    for deal_id in range(12):
        deal = Deal(RULES, dealer=deal_id % 4, hands=deal_hands(rng, RULES), deal_id=deal_id)
        tracked = TrackedDeal(deal)
        while deal.phase in (Phase.AUCTION, Phase.PLAY):
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            action = composed.act(info, SPACE, mask)
            if info.phase == Phase.AUCTION:
                assert action == bidder.act(info, SPACE, mask)
                n_bid += 1
            else:
                assert action == reference_player.act(info, SPACE, mask)
                n_play += 1
            tracked.step(SPACE.decode(action))
    assert n_bid > 0 and n_play > 0
