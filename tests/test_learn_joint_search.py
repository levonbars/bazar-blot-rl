"""Joint bid search (`learn/joint_search.py`), forced decisions, the auction lambda, and the
8-hearts support probe."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.nn.network import CentralCritic, build_network  # noqa: E402
from bazarblot.agents.nn.ppo_agent import NNAgent, decide, recompute_log_prob  # noqa: E402
from bazarblot.core.auction import PassAction  # noqa: E402
from bazarblot.core.deal import Deal, Phase  # noqa: E402
from bazarblot.core.dealing import deal_hands  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space, legal_mask  # noqa: E402
from bazarblot.env.infoset import info_set  # noqa: E402
from bazarblot.env.tracked_deal import TrackedDeal  # noqa: E402
from bazarblot.learn.joint_search import JointSearch, JointSearchConfig  # noqa: E402
from bazarblot.learn.rollout import Transition, _gae, play_rollout_deals  # noqa: E402
from bazarblot.learn.scenarios import heart_support  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def _net():  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    net = build_network(RULES, hidden=(32,))
    with torch.no_grad():  # open at the top of the ladder so greedy auctions end quickly
        net.delta_head.bias.copy_(torch.arange(net.delta_head.bias.numel(), dtype=torch.float32))
    net.eval()
    return net


def test_forced_action_is_taken_and_scored_by_the_network() -> None:
    net = _net()
    rng = random.Random(0)
    deal = Deal(RULES, dealer=0, hands=deal_hands(rng, RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, deal.to_act, (0, 0), 0)
    mask = legal_mask(info, SPACE)
    target = SPACE.bid_index(9, "H", False)
    d = decide(net, info, SPACE, mask, sample=True, force_env_action=target)
    assert d.env_action_idx == target and d.forced
    lp, _, _ = recompute_log_prob(
        net, SPACE, torch.from_numpy(d.obs), d.mask, d.phase, d.base_level, d.action_idx
    )
    assert float(lp) == pytest.approx(d.log_prob, abs=1e-5)


def test_auction_lambda_zero_judges_a_bid_by_the_next_position() -> None:
    from bazarblot.agents.nn.ppo_agent import Decision

    def tr(phase: Phase, value: float, reward: float = 0.0) -> Transition:
        dec = Decision(0, 0.0, value, None, None, phase, 0)  # type: ignore[arg-type]
        return Transition(decision=dec, value=value, reward=reward)

    traj = [
        tr(Phase.AUCTION, 0.1),
        tr(Phase.AUCTION, 0.3),
        tr(Phase.PLAY, 0.5),
        tr(Phase.PLAY, 0.2, 1.0),
    ]
    _gae(traj, gamma=1.0, lam=0.95, lam_auction=0.0)
    assert traj[0].advantage == pytest.approx(0.3 - 0.1)
    assert traj[1].advantage == pytest.approx(0.5 - 0.3)
    # card play keeps the ordinary lambda
    assert traj[2].advantage == pytest.approx((0.2 - 0.5) + 0.95 * (1.0 - 0.2))


@pytest.mark.parametrize("evaluate", ["playout", "critic"])
def test_search_returns_legal_bids(evaluate: str) -> None:
    net = _net()
    critic = CentralCritic(hidden=(32,))
    js = JointSearch(RULES, SPACE, net, critic, JointSearchConfig(deltas=(1,), evaluate=evaluate))
    rng = random.Random(1)
    agent = NNAgent(net)
    checked = 0
    for d in range(4):
        deal = Deal(RULES, dealer=d % 4, hands=deal_hands(rng, RULES), deal_id=d)
        tracked = TrackedDeal(deal)
        while deal.phase == Phase.AUCTION and len(tracked.auction_log) < 12:
            info = info_set(tracked, deal.to_act, (0, 0), 0)
            mask = legal_mask(info, SPACE)
            for found in (js.search(tracked), js.best_reply(tracked)):
                assert found is None or bool(mask[found])
            checked += 1
            tracked.step(SPACE.decode(agent.act(info, SPACE, mask)))
    assert checked > 0 and js.stats.searches == checked


def test_rollout_with_search_records_forced_bids() -> None:
    net = _net()
    trajectories, stats = play_rollout_deals(
        RULES,
        SPACE,
        net,
        range(6),
        3,
        random.Random(0),
        reward_scale=96.0,
        search_config=JointSearchConfig(prob=1.0, deltas=(1,)),
    )
    assert stats.n_searches > 0
    forced = [t for traj in trajectories for t in traj if t.decision.forced]
    assert len(forced) == stats.n_search_deviations + stats.n_search_reply_changes


def test_heart_support_probe() -> None:
    net = _net()
    report = heart_support(RULES, SPACE, NNAgent(net), n_deals=20)
    for rates in (report.fit, report.no_fit):
        assert sum(rates.values()) == pytest.approx(1.0)
    assert "fit:" in report.summary()


def test_pass_is_always_a_candidate() -> None:
    net = _net()
    js = JointSearch(RULES, SPACE, net, None, JointSearchConfig())
    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(2), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    tracked.step(PassAction())
    assert 0 in js._candidates(tracked, include=0)
