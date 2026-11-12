"""Does the auction carry information to the partner? (roadmap M7 done-when, item 3; spec §9.)

Measured on greedy self-play deals, at each seat's FIRST card-play decision, by comparing the
network's real observation with a counterfactual one in which partner's auction actions are
replaced by passes (same seats, same order). Everything else is unchanged, including the final
contract blocks, so the difference is information in partner's bids BEYOND the final contract.

- `partner_info_gain`: mean over partner's unseen cards of `log p(partner holds it)` from the
  belief head, real minus silenced, in nats per card. 0 = partner's bids tell it nothing. The
  belief head is trained on ground truth, so this measures what the bids CAN tell a seat.
- `card_change_rate`: how often the greedy first card differs between the two observations, i.e.
  whether the PLAY actually uses partner's bids. An untrained network gives a few percent from
  noise alone, so compare against `untrained_card_change_rate` from a control network.

Confidence intervals are bootstrapped over deals (the four seats of one deal are not independent).
This is a probe of the network, not a training signal: it never feeds back into training.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass

import numpy as np
import torch

from bazarblot.agents.nn.network import PolicyValueNet
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.core.auction import PassAction
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.infoset import InfoSet, info_set
from bazarblot.env.obs import encode, flatten
from bazarblot.env.tracked_deal import AuctionEvent, TrackedDeal
from bazarblot.eval.metrics import ConfidenceInterval, bootstrap_ci
from bazarblot.learn.belief import belief_target

_PARTNER = 1  # belief-head index of rel seat 2 (index = rel - 1)


@dataclass(frozen=True, slots=True)
class SignalingReport:
    n_deals: int
    n_decisions: int
    partner_info_gain: ConfidenceInterval
    card_change_rate: ConfidenceInterval
    partner_declared_share: float

    def summary(self) -> str:
        g, c = self.partner_info_gain, self.card_change_rate
        return (
            f"partner_info_gain={g.mean:+.4f} nats/card CI ({g.lo:+.4f}, {g.hi:+.4f})  "
            f"card_change_rate={c.mean:.3f} CI ({c.lo:.3f}, {c.hi:.3f})  "
            f"(n={self.n_decisions} decisions in {self.n_deals} deals)"
        )


def silence_partner(info: InfoSet) -> InfoSet:
    """`info` with every auction action of the deciding seat's partner replaced by a pass."""
    partner = (info.seat + 2) % 4
    log = tuple(
        AuctionEvent(e.seat, PassAction()) if e.seat == partner else e for e in info.auction_log
    )
    return dataclasses.replace(info, auction_log=log)


def measure_signaling(
    rules: RuleConfig,
    space: ActionSpace,
    net: PolicyValueNet,
    n_deals: int = 400,
    seed: int = 5,
    n_resamples: int = 2000,
) -> SignalingReport | None:
    """Play `n_deals` greedy self-play deals with `net` in all four seats and probe each seat's
    first card-play decision. Returns `None` if no deal reached card play (e.g. a network that
    always passes)."""
    was_training = net.training
    net.eval()
    agent = NNAgent(net, sample=False)
    rng = random.Random(seed)
    gains: list[float] = []
    changes: list[float] = []
    n_decisions = 0
    n_partner_declared = 0
    try:
        for d in range(n_deals):
            deal = Deal(rules, dealer=d % 4, hands=deal_hands(rng, rules), deal_id=d)
            tracked = TrackedDeal(deal)
            probed: set[int] = set()
            deal_gain: list[float] = []
            deal_change: list[float] = []
            while deal.phase in (Phase.AUCTION, Phase.PLAY):
                if (
                    deal.phase == Phase.AUCTION
                    and len(tracked.auction_log) >= rules.auction.max_auction_steps
                ):
                    break  # circling auction (greedy, untrained); no card play to probe
                seat = deal.to_act
                info = info_set(tracked, seat, match_score=(0, 0), deal_number=d)
                mask = legal_mask(info, space)
                if deal.phase == Phase.PLAY and seat not in probed:
                    probed.add(seat)
                    gain, changed = _probe(net, space, info, mask, deal)
                    if gain is not None:
                        deal_gain.append(gain)
                    deal_change.append(float(changed))
                    assert deal.contract is not None
                    n_partner_declared += int(deal.contract.declarer_seat == (seat + 2) % 4)
                    n_decisions += 1
                tracked.step(space.decode(agent.act(info, space, mask)))
            if deal_change:
                changes.append(float(np.mean(deal_change)))
            if deal_gain:
                gains.append(float(np.mean(deal_gain)))
    finally:
        net.train(was_training)
    if not changes or not gains:
        return None
    return SignalingReport(
        n_deals=len(changes),
        n_decisions=n_decisions,
        partner_info_gain=bootstrap_ci(gains, n_resamples=n_resamples, seed=seed),
        card_change_rate=bootstrap_ci(changes, n_resamples=n_resamples, seed=seed),
        partner_declared_share=n_partner_declared / max(1, n_decisions),
    )


def _probe(
    net: PolicyValueNet,
    space: ActionSpace,
    info: InfoSet,
    mask: np.ndarray,  # type: ignore[type-arg]
    deal: Deal,
) -> tuple[float | None, bool]:
    target = belief_target(deal.hands, info.seat)
    partner_cards = np.nonzero(target == _PARTNER)[0]
    obs = torch.from_numpy(
        np.stack([flatten(encode(info)), flatten(encode(silence_partner(info)))])
    )
    with torch.no_grad():
        out = net(obs)
    log_p = torch.log_softmax(out["belief"], dim=-1).numpy()  # (2, 32, 3)
    gain: float | None = None
    if len(partner_cards):
        real = log_p[0, partner_cards, _PARTNER].mean()
        silenced = log_p[1, partner_cards, _PARTNER].mean()
        gain = float(real - silenced)
    card_mask = torch.from_numpy(mask[space.play_start :])
    picks = out["card"].masked_fill(~card_mask, float("-inf")).argmax(dim=-1)
    return gain, bool(picks[0] != picks[1])


@dataclass(frozen=True, slots=True)
class PartnerSupportReport:
    """What a seat does when PARTNER holds the standing bid and a bid is legal, split by how many
    aces the seat holds. The convention the project owner described (answer partner's bid with
    aces, +1 or +2, before bidding one's own suit) would show as support rising with aces and raise
    sizes other than +1; agents that bid as individuals show flat support and own-suit bids."""

    n_decisions: int
    by_aces: dict[int, tuple[int, float, float, float]]
    """aces held (3 = 3 or more) -> (n, pass rate, support-partner's-suit rate, other-suit rate)"""
    support_raise_sizes: dict[int, int]

    @property
    def support_gap(self) -> float | None:
        """Support rate with 2+ aces minus with 0 aces: ~0 means support ignores aces."""
        lo = self.by_aces.get(0)
        hi = [v for k, v in self.by_aces.items() if k >= 2]
        if lo is None or lo[0] == 0 or not hi or sum(v[0] for v in hi) == 0:
            return None
        n_hi = sum(v[0] for v in hi)
        return sum(v[0] * v[2] for v in hi) / n_hi - lo[2]

    def summary(self) -> str:
        cells = "  ".join(
            f"{k}{'+' if k == 3 else ''}A: n={v[0]} sup {v[2]:.2f} own {v[3]:.2f}"
            for k, v in sorted(self.by_aces.items())
        )
        gap = self.support_gap
        gap_s = f"{gap:+.3f}" if gap is not None else "n/a"
        return f"{cells}  | support gap(2+A - 0A) {gap_s}  raise sizes {self.support_raise_sizes}"


def measure_partner_support(
    rules: RuleConfig,
    space: ActionSpace,
    net: PolicyValueNet,
    n_deals: int = 400,
    seed: int = 9,
) -> PartnerSupportReport:
    """Greedy self-play with `net` in all four seats; see `PartnerSupportReport`."""
    from bazarblot.core.auction import BidAction
    from bazarblot.core.cards import rank_of

    was_training = net.training
    net.eval()
    agent = NNAgent(net, sample=False)
    rng = random.Random(seed)
    counts: dict[int, list[int]] = {}  # aces -> [n, pass, support, other]
    sizes: dict[int, int] = {}
    try:
        for d in range(n_deals):
            deal = Deal(rules, dealer=d % 4, hands=deal_hands(rng, rules), deal_id=d)
            tracked = TrackedDeal(deal)
            while deal.phase == Phase.AUCTION:
                if len(tracked.auction_log) >= rules.auction.max_auction_steps:
                    break
                info = info_set(tracked, deal.to_act, match_score=(0, 0), deal_number=d)
                mask = legal_mask(info, space)
                action_idx = agent.act(info, space, mask)
                standing = info.auction_state.standing_bid
                if (
                    standing is not None
                    and standing.seat == (info.seat + 2) % 4
                    and mask[space.bid_start : space.play_start].any()
                ):
                    aces = min(3, sum(1 for c in info.hand if rank_of(c) == "A"))
                    row = counts.setdefault(aces, [0, 0, 0, 0])
                    row[0] += 1
                    action = space.decode(action_idx)
                    if isinstance(action, BidAction):
                        if action.contract_type == standing.contract_type:
                            row[2] += 1
                            size = action.level - standing.level
                            sizes[size] = sizes.get(size, 0) + 1
                        else:
                            row[3] += 1
                    else:
                        row[1] += 1
                tracked.step(space.decode(action_idx))
    finally:
        net.train(was_training)
    by_aces = {
        k: (v[0], v[1] / v[0], v[2] / v[0], v[3] / v[0]) for k, v in counts.items() if v[0] > 0
    }
    return PartnerSupportReport(
        n_decisions=sum(v[0] for v in counts.values()),
        by_aces=by_aces,
        support_raise_sizes=dict(sorted(sizes.items())),
    )
