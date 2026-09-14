"""M7: `NNAgent`'s masked hierarchical sampling -- the two properties that matter most before any
training loop is worth trusting: (1) it only ever plays legal actions, on both an untrained
(random-weight) network and across many random deals/seats, and (2) `recompute_log_prob` exactly
reproduces `decide()`'s own log-probability when the network hasn't changed -- the PPO ratio at
the very first gradient step must be `exp(0) == 1` for any minibatch drawn from the rollout that
produced it, and this is the property that guarantees it."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.nn.network import build_network  # noqa: E402
from bazarblot.agents.nn.ppo_agent import NNAgent, decide, recompute_log_prob  # noqa: E402
from bazarblot.core.deal import Deal, Phase  # noqa: E402
from bazarblot.core.dealing import deal_hands  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space, legal_mask  # noqa: E402
from bazarblot.env.infoset import info_set  # noqa: E402
from bazarblot.env.tracked_deal import TrackedDeal  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def test_nn_agent_only_ever_plays_legal_actions() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    agent = NNAgent(net, sample=True)
    rng = random.Random(0)
    for deal_id in range(15):
        deal = Deal(RULES, dealer=deal_id % 4, hands=deal_hands(rng, RULES), deal_id=deal_id)
        tracked = TrackedDeal(deal)
        while deal.phase in (Phase.AUCTION, Phase.PLAY):
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            action_idx = agent.act(info, SPACE, mask)
            assert mask[action_idx], f"NNAgent chose an illegal action at phase {info.phase}"
            tracked.step(SPACE.decode(action_idx))


def test_forced_bid_prob_overrides_meta_choice_toward_bid() -> None:
    """`forced_bid_prob=1.0` must make the agent choose BID at every auction decision where any
    bid is legal, regardless of what the (untrained, near-uniform) network itself would have
    sampled -- the mechanism `learn/train.py` anneals down over training to fix the pass-collapse
    (`docs/03-implementation-roadmap.md` M7)."""
    torch.manual_seed(0)
    net = build_network(RULES)
    rng = random.Random(0)
    n_bid_decisions = 0
    for deal_id in range(10):
        deal = Deal(RULES, dealer=deal_id % 4, hands=deal_hands(rng, RULES), deal_id=deal_id)
        tracked = TrackedDeal(deal)
        while deal.phase == Phase.AUCTION:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            any_bid_legal = bool(mask[SPACE.bid_start : SPACE.play_start].any())
            decision = decide(net, info, SPACE, mask, sample=True, forced_bid_prob=1.0)
            if any_bid_legal:
                assert SPACE.bid_start <= decision.action_idx < SPACE.play_start
                n_bid_decisions += 1
            tracked.step(SPACE.decode(decision.action_idx))
        while deal.phase == Phase.PLAY:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            decision = decide(net, info, SPACE, mask, sample=True, forced_bid_prob=1.0)
            tracked.step(SPACE.decode(decision.action_idx))
    assert n_bid_decisions > 0


def test_forced_bid_log_prob_still_matches_recompute() -> None:
    """The critical correctness property: even when `forced_bid_prob` overrides WHICH action
    gets sampled, the returned `log_prob` must still be the network's own TRUE probability of
    that action -- exactly what `recompute_log_prob` would compute -- since PPO's importance
    ratio depends on this being accurate, not on how the action was actually generated."""
    torch.manual_seed(1)
    net = build_network(RULES)
    net.eval()
    rng = random.Random(1)
    checked = 0
    for deal_id in range(15):
        deal = Deal(RULES, dealer=deal_id % 4, hands=deal_hands(rng, RULES), deal_id=deal_id)
        tracked = TrackedDeal(deal)
        while deal.phase == Phase.AUCTION:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            decision = decide(net, info, SPACE, mask, sample=True, forced_bid_prob=1.0)
            obs_t = torch.from_numpy(decision.obs)
            log_prob, _entropy, value = recompute_log_prob(
                net,
                SPACE,
                obs_t,
                decision.mask,
                decision.phase,
                decision.base_level,
                decision.action_idx,
            )
            assert log_prob.item() == pytest.approx(decision.log_prob, abs=1e-5)
            assert value.item() == pytest.approx(decision.value, abs=1e-5)
            checked += 1
            tracked.step(SPACE.decode(decision.action_idx))
        while deal.phase == Phase.PLAY:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            decision = decide(net, info, SPACE, mask, sample=True)
            tracked.step(SPACE.decode(decision.action_idx))
    assert checked > 0


def test_recompute_log_prob_matches_decide_when_network_unchanged() -> None:
    torch.manual_seed(1)
    net = build_network(RULES)
    net.eval()
    rng = random.Random(1)
    checked_auction = checked_play = 0
    for deal_id in range(20):
        deal = Deal(RULES, dealer=deal_id % 4, hands=deal_hands(rng, RULES), deal_id=deal_id)
        tracked = TrackedDeal(deal)
        while deal.phase in (Phase.AUCTION, Phase.PLAY):
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=deal_id)
            mask = legal_mask(info, SPACE)
            decision = decide(net, info, SPACE, mask, sample=True)
            obs_t = torch.from_numpy(decision.obs)
            log_prob, _entropy, value = recompute_log_prob(
                net,
                SPACE,
                obs_t,
                decision.mask,
                decision.phase,
                decision.base_level,
                decision.action_idx,
            )
            assert log_prob.item() == pytest.approx(decision.log_prob, abs=1e-5)
            assert value.item() == pytest.approx(decision.value, abs=1e-5)
            if decision.phase == Phase.AUCTION:
                checked_auction += 1
            else:
                checked_play += 1
            tracked.step(SPACE.decode(decision.action_idx))
    assert checked_auction > 0
    assert checked_play > 0
