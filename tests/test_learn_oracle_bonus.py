"""The oracle input (partner's hand, training only), the information bonus for auction actions,
and the oracle match."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.nn.network import (  # noqa: E402
    OBS_DIM,
    ORACLE_DIM,
    CentralCritic,
    build_network,
)
from bazarblot.agents.nn.ppo_agent import decide, recompute_log_prob  # noqa: E402
from bazarblot.core.deal import Deal, Phase  # noqa: E402
from bazarblot.core.dealing import deal_hands  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space, legal_mask  # noqa: E402
from bazarblot.env.infoset import info_set  # noqa: E402
from bazarblot.env.tracked_deal import TrackedDeal  # noqa: E402
from bazarblot.learn.checkpoint import load_actor_state  # noqa: E402
from bazarblot.learn.oracle_eval import oracle_match  # noqa: E402
from bazarblot.learn.rollout import play_rollout_deals  # noqa: E402
from bazarblot.learn.train import _faded  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def _net():  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    net = build_network(RULES, hidden=(32,))
    with torch.no_grad():  # open at the top of the ladder so greedy auctions end quickly
        net.delta_head.bias.copy_(torch.arange(net.delta_head.bias.numel(), dtype=torch.float32))
    net.eval()
    return net


def _first_info():  # type: ignore[no-untyped-def]
    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(0), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, deal.to_act, (0, 0), 0)
    return deal, info, legal_mask(info, SPACE)


def test_untrained_oracle_input_changes_nothing() -> None:
    net = _net()
    deal, info, mask = _first_info()
    plain = decide(net, info, SPACE, mask, sample=False)
    seen = decide(
        net, info, SPACE, mask, sample=False, oracle_cards=deal.hands[(info.seat + 2) % 4]
    )
    assert plain.obs.shape == (OBS_DIM,) and seen.obs.shape == (OBS_DIM + ORACLE_DIM,)
    assert seen.obs[OBS_DIM:].sum() == 8
    assert seen.log_prob == pytest.approx(plain.log_prob, abs=1e-6)
    assert seen.value == pytest.approx(plain.value, abs=1e-6)


def test_oracle_input_reaches_the_policy_and_is_rescored_consistently() -> None:
    net = _net()
    with torch.no_grad():
        net.oracle_weight.normal_()
    deal, info, mask = _first_info()
    d = decide(net, info, SPACE, mask, sample=True, oracle_cards=deal.hands[(info.seat + 2) % 4])
    plain = decide(net, info, SPACE, mask, sample=False)
    assert d.value != pytest.approx(plain.value, abs=1e-6)
    lp, _, _ = recompute_log_prob(
        net, SPACE, torch.from_numpy(d.obs), d.mask, d.phase, d.base_level, d.action_idx
    )
    assert float(lp) == pytest.approx(d.log_prob, abs=1e-5)
    # the critic ignores the oracle columns
    critic = CentralCritic(hidden=(16,))
    priv = torch.zeros(1, 96)
    v1 = critic(torch.from_numpy(d.obs).unsqueeze(0), priv)
    v2 = critic(torch.from_numpy(d.obs[:OBS_DIM]).unsqueeze(0), priv)
    assert float(v1) == pytest.approx(float(v2))


def test_old_checkpoint_state_loads_without_oracle_weights() -> None:
    net = _net()
    state = {k: v for k, v in net.state_dict().items() if k != "oracle_weight"}
    fresh = build_network(RULES, hidden=(32,))
    load_actor_state(fresh, state)
    assert float(fresh.oracle_weight.abs().sum()) == 0.0
    with pytest.raises(RuntimeError):
        load_actor_state(fresh, {**state, "bogus": torch.zeros(1)})


def test_rollout_with_oracle_and_info_bonus() -> None:
    net = _net()
    trajectories, stats = play_rollout_deals(
        RULES,
        SPACE,
        net,
        range(8),
        5,
        random.Random(0),
        reward_scale=96.0,
        oracle_keep=1.0,
        info_bonus=0.5,
    )
    flat = [t for traj in trajectories for t in traj]
    assert flat and all(t.decision.obs.shape == (OBS_DIM + ORACLE_DIM,) for t in flat)
    assert all(
        t.decision.obs[OBS_DIM:].sum() > 0 for t in flat if t.decision.phase == Phase.AUCTION
    )
    assert stats.info_gain_n > 0
    bonuses = [t.bonus for t in flat if t.decision.phase == Phase.AUCTION]
    assert any(b != 0.0 for b in bonuses)
    assert all(t.bonus == 0.0 for t in flat if t.decision.phase == Phase.PLAY)
    assert stats.info_gain_sum == pytest.approx(sum(bonuses) / 0.5)


def test_rollout_without_the_options_is_unchanged() -> None:
    net = _net()
    torch.manual_seed(1)
    a, _ = play_rollout_deals(RULES, SPACE, net, range(4), 5, random.Random(0), reward_scale=96.0)
    torch.manual_seed(1)
    b, _ = play_rollout_deals(
        RULES, SPACE, net, range(4), 5, random.Random(0), reward_scale=96.0, info_bonus=0.0
    )
    assert [t.reward for tr in a for t in tr] == [t.reward for tr in b for t in tr]
    assert all(t.decision.obs.shape == (OBS_DIM,) for tr in a for t in tr)


def test_fade_schedule() -> None:
    assert _faded(1.0, 0, (10, 20)) == 1.0
    assert _faded(1.0, 15, (10, 20)) == pytest.approx(0.5)
    assert _faded(1.0, 25, (10, 20)) == 0.0
    assert _faded(0.3, 5, (0, 0)) == 0.0


def test_oracle_match_runs() -> None:
    net = _net()
    report = oracle_match(RULES, SPACE, net, _net(), n_pairs=3)
    assert report.n_pairs == 3
    assert "margin=" in report.summary()
