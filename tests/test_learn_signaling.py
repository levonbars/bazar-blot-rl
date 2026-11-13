"""The signaling probe (`learn/signaling.py`): the counterfactual must hide exactly partner's
bids, and the report must be well-formed on a network that reaches card play."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.nn.network import apply_action_prior, build_network  # noqa: E402
from bazarblot.core.auction import PassAction  # noqa: E402
from bazarblot.core.deal import Deal, Phase  # noqa: E402
from bazarblot.core.dealing import deal_hands  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space, legal_mask  # noqa: E402
from bazarblot.env.infoset import info_set  # noqa: E402
from bazarblot.env.tracked_deal import TrackedDeal  # noqa: E402
from bazarblot.learn.signaling import measure_signaling, silence_partner  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def test_silence_partner_replaces_only_partners_actions() -> None:
    rng = random.Random(0)
    for seed in range(30):
        deal = Deal(RULES, dealer=seed % 4, hands=deal_hands(rng, RULES), deal_id=seed)
        tracked = TrackedDeal(deal)
        while deal.phase == Phase.AUCTION:
            info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=0)
            legal = legal_mask(info, SPACE).nonzero()[0]
            tracked.step(SPACE.decode(int(rng.choice(list(legal[:12])))))
        info = info_set(tracked, seed % 4, match_score=(0, 0), deal_number=0)
        quiet = silence_partner(info)
        partner = (info.seat + 2) % 4
        assert len(quiet.auction_log) == len(info.auction_log)
        for orig, new in zip(info.auction_log, quiet.auction_log, strict=True):
            assert new.seat == orig.seat
            if orig.seat == partner:
                assert isinstance(new.action, PassAction)
            else:
                assert new.action == orig.action


def test_measure_signaling_report_is_well_formed() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    apply_action_prior(net, raise_decay=0.5, capot_bias=-4.0, double_bias=-3.0)
    with torch.no_grad():  # greedy untrained auctions must end: open at the top of the ladder,
        # where nobody can bid over it, so every deal reaches card play
        net.delta_head.bias.copy_(torch.arange(net.delta_head.bias.numel(), dtype=torch.float32))
    report = measure_signaling(RULES, SPACE, net, n_deals=60, seed=1, n_resamples=200)
    assert report is not None
    assert report.n_decisions > 0
    g, c = report.partner_info_gain, report.card_change_rate
    assert g.lo <= g.mean <= g.hi
    assert 0.0 <= c.lo <= c.mean <= c.hi <= 1.0
    assert 0.0 <= report.partner_declared_share <= 1.0
    assert net.training  # restored


def test_partner_support_report_is_consistent() -> None:
    from bazarblot.learn.signaling import measure_partner_support

    torch.manual_seed(0)
    net = build_network(RULES)
    apply_action_prior(net, raise_decay=0.5, capot_bias=-4.0, double_bias=-3.0)
    report = measure_partner_support(RULES, SPACE, net, n_deals=80, seed=2)
    assert report.n_decisions == sum(v[0] for v in report.by_aces.values())
    for n, p_pass, p_sup, p_own in report.by_aces.values():
        assert n > 0 and abs(p_pass + p_sup + p_own - 1.0) < 1e-9
    assert all(size >= 1 for size in report.support_raise_sizes)
    assert isinstance(report.summary(), str)
