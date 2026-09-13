"""M4: `HeuristicAgent`, the "club player" baseline.

`test_heuristic_beats_random_by_a_wide_margin` is the roadmap's stated M4 "done when" bar for
this agent. Note this is a paired-deal-free comparison (unpaired, independent seeds) — the full
paired/duplicate evaluation harness `docs/03-implementation-roadmap.md` describes is M5's
deliverable, not built yet. At this margin (see the test's own assertion threshold) the
distinction doesn't matter: heuristic wins by so much that per-deal variance can't explain it
even unpaired, but a real paper-quality comparison would still route through M5's harness once
it exists — flagged explicitly in the M4 status write-up, not silently substituted for it.
"""

from __future__ import annotations

import random

from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.auction import BidAction, PassAction
from bazarblot.core.deal import Deal
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space, legal_mask
from bazarblot.env.aec import BazarBlotAEC
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal

RULES = load_default()


def _run_deal(env: BazarBlotAEC, agents: dict[int, object], seed: int) -> dict[str, float] | None:
    env.reset(seed=seed)
    steps = 0
    last_rewards: dict[str, float] | None = None
    while env.agents and steps < 300:
        a = env.agent_selection
        if env.terminations[a] or env.truncations[a]:
            env.step(None)
            steps += 1
            continue
        seat = int(a.split("_")[1])
        info = env._info_for(seat)
        mask = env.observe(a)["action_mask"].astype(bool)
        action = agents[seat].act(info, env.action_space_obj, mask)  # type: ignore[attr-defined]
        assert mask[action], f"agent for seat {seat} chose an illegal action"
        env.step(action)
        if any(env.rewards.values()):
            last_rewards = dict(env.rewards)
        steps += 1
    return last_rewards


def test_heuristic_only_ever_plays_legal_actions() -> None:
    env = BazarBlotAEC(episode_unit="deal")
    h = HeuristicAgent(RULES, random.Random(1))
    agents = {s: h for s in range(4)}
    for seed in range(60):
        _run_deal(env, agents, seed)


def test_heuristic_never_raises_over_its_own_partner() -> None:
    """A documented simplification (no bidding-convention model) — pin the behavior it implies:
    the heuristic never competes against a standing bid from its own team.

    Auction order rotates one seat at a time, so after the opener bids and the next seat passes,
    it's exactly the opener's partner's turn — a fixed two-step sequence, not a search."""
    space = build_action_space(RULES)
    h = HeuristicAgent(RULES, random.Random(2))
    for seed in range(100):
        rng = random.Random(seed)
        hands = deal_hands(rng, RULES)
        deal = Deal(RULES, dealer=seed % 4, hands=hands, deal_id=seed)
        tracked = TrackedDeal(deal)

        opener = deal.to_act
        tracked.step(BidAction(level=RULES.auction.min_bid, contract_type="H", capot=False))
        tracked.step(PassAction())  # the seat between opener and their partner
        partner_seat = (opener + 2) % 4
        assert deal.to_act == partner_seat

        info = info_set(tracked, partner_seat, match_score=(0, 0), deal_number=0)
        mask = legal_mask(info, space)
        action = h.act(info, space, mask)
        assert action == 0  # PASS


def test_heuristic_beats_random_by_a_wide_margin() -> None:
    """The M4 done-when bar: heuristic (seats 0, 2) vs. random (seats 1, 3), squashed per-deal
    margin (spec §6's `tanh(delta/24)`, already what `env/aec.py` reports). A margin this
    lopsided (see the assertion) can't be explained by deal variance even without pairing."""
    env = BazarBlotAEC(episode_unit="deal")
    h = HeuristicAgent(RULES, random.Random(10))
    r = RandomAgent(random.Random(11))
    agents = {0: h, 2: h, 1: r, 3: r}

    n = 200
    total = 0.0
    decided = 0
    for seed in range(n):
        rewards = _run_deal(env, agents, seed)
        if rewards is None:
            continue
        decided += 1
        total += rewards["player_0"]
        assert rewards["player_0"] == rewards["player_2"]
        assert rewards["player_1"] == rewards["player_3"]

    assert decided > n * 0.9  # nearly every deal should resolve within the step cap
    mean_margin = total / decided
    assert mean_margin > 0.5, f"heuristic's mean squashed margin over random was only {mean_margin}"
