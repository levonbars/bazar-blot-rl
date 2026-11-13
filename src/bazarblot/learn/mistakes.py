"""Counts of obvious mistakes in greedy self-play (the same agent in all four seats).

Beating the heuristic on average hid basic errors a club player would never make. These counts
make them visible in every evaluation. Each rate is over the decisions where the mistake was
possible AND a better legal option existed:

- `failed`: contracts that fail.
- `bid_other_suit_over_partner`: bids over partner's standing bid in a different suit.
- `takes_partner_trick`: last to play, partner already winning, plays a card that wins the trick
  when a card that leaves it with partner was legal.
- `ducks_valuable_trick`: last to play, opponents winning a trick worth 10+ points, plays a
  losing card when a winning card was legal.
- `throws_points`: last to play, discarding to an opponents' trick, throws an Ace or a Ten when a
  card worth less than 10 was legal.
- `endgame_loss`: with 3 or fewer cards each, plays a card that loses points against the best
  card, by an exact solve with all hands known (with so few cards left most of them are
  inferable, so this slightly overstates real mistakes; the heuristic scores ~19%).

A probe for evaluation only; it never feeds back into training.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from bazarblot.agents.base import Agent
from bazarblot.core.auction import BidAction
from bazarblot.core.cards import TEAM_OF
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.play import PlayCardAction, current_winner
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal
from bazarblot.solver.dd import Solver

_RATES = (
    ("failed", "contracts", "contracts failed"),
    ("bid_other_suit", "bid_over_partner", "other suit over partner"),
    ("takes_partner_trick", "partner_winning_last", "takes partner's trick"),
    ("ducks_valuable_trick", "can_win_valuable_last", "ducks a 10+ trick"),
    ("throws_points", "discards_to_opps_last", "throws A/10"),
    ("endgame_loss", "endgame_decisions", "endgame loses points"),
)


@dataclass(frozen=True, slots=True)
class MistakeReport:
    n_deals: int
    counts: dict[str, int]
    endgame_points_lost: float  # mean raw points lost per endgame decision

    def rate(self, name: str) -> float | None:
        for num, den, _ in _RATES:
            if num == name:
                return self.counts.get(num, 0) / self.counts[den] if self.counts.get(den) else None
        raise KeyError(name)

    def summary(self) -> str:
        parts = []
        for num, den, label in _RATES:
            r = self.rate(num)
            n = self.counts.get(den, 0)
            parts.append(f"{label} {r:.3f} (n={n})" if r is not None else f"{label} n/a")
        return "  ".join(parts) + f"  endgame pts lost/decision {self.endgame_points_lost:.2f}"


def _legal_cards(deal: Deal) -> list[int]:
    return [a.card for a in deal.legal_actions() if isinstance(a, PlayCardAction)]


def _endgame_values(deal: Deal, seat: int, legal: list[int], rules: RuleConfig) -> dict[int, int]:
    """Exact points (incl. last-trick bonus) `seat`'s team takes from here for each legal card."""
    assert deal.contract is not None and deal.tables is not None
    t = deal.tables
    team = TEAM_OF[seat]
    solver = Solver(deal.contract.contract_type, team, rules)
    trick = list(deal.current_trick)
    values = {}
    for card in legal:
        hands = list(deal.hands)
        hands[seat] = hands[seat] - {card}
        after = (hands[0], hands[1], hands[2], hands[3])
        new_trick = (*trick, (seat, card))
        if len(new_trick) == 4:
            winner = current_winner(new_trick, t)
            gain = sum(t.points[c] for _, c in new_trick)
            last = all(len(h) == 0 for h in hands)
            if last:
                gain += rules.contracts.last_hand_bonus
            rest = 0 if last else solver.solve_from(after, winner, ()).declarer_points
            values[card] = (gain if TEAM_OF[winner] == team else 0) + rest
        else:
            nxt = (seat + 1) % 4
            values[card] = solver.solve_from(after, nxt, new_trick).declarer_points
    return values


def measure_mistakes(
    rules: RuleConfig, space: ActionSpace, agent: Agent, n_deals: int = 200, seed: int = 31337
) -> MistakeReport:
    rng = random.Random(seed)
    c: Counter[str] = Counter()
    endgame_lost = 0
    for d in range(n_deals):
        deal = Deal(rules, dealer=d % 4, hands=deal_hands(rng, rules), deal_id=d)
        tracked = TrackedDeal(deal)
        while deal.phase in (Phase.AUCTION, Phase.PLAY):
            if (
                deal.phase == Phase.AUCTION
                and len(tracked.auction_log) >= rules.auction.max_auction_steps - 1
            ):
                break
            seat = deal.to_act
            info = info_set(tracked, seat, match_score=(0, 0), deal_number=d)
            action = space.decode(agent.act(info, space, legal_mask(info, space)))
            if deal.phase == Phase.AUCTION and isinstance(action, BidAction):
                standing = deal.auction_state.standing_bid
                if standing is not None and standing.seat == (seat + 2) % 4:
                    c["bid_over_partner"] += 1
                    c["bid_other_suit"] += action.contract_type != standing.contract_type
            elif deal.phase == Phase.PLAY:
                assert isinstance(action, PlayCardAction) and deal.tables is not None
                _count_play(deal, seat, action.card, rules, c)
                legal = _legal_cards(deal)
                if len(deal.hands[seat]) <= 3 and len(legal) > 1:
                    values = _endgame_values(deal, seat, legal, rules)
                    loss = max(values.values()) - values[action.card]
                    c["endgame_decisions"] += 1
                    c["endgame_loss"] += loss > 0
                    endgame_lost += loss
            tracked.step(action)
        if deal.result is not None:
            c["contracts"] += 1
            c["failed"] += not deal.result.made
    return MistakeReport(
        n_deals=n_deals,
        counts=dict(c),
        endgame_points_lost=endgame_lost / max(1, c["endgame_decisions"]),
    )


def _count_play(deal: Deal, seat: int, card: int, rules: RuleConfig, c: Counter[str]) -> None:
    trick = list(deal.current_trick)
    if len(trick) != 3:
        return
    t = deal.tables
    assert t is not None
    legal = _legal_cards(deal)
    if len(legal) < 2:
        return
    wins = {x: current_winner([*trick, (seat, x)], t) == seat for x in legal}
    if current_winner(trick, t) == (seat + 2) % 4:
        if not all(wins.values()):
            c["partner_winning_last"] += 1
            c["takes_partner_trick"] += wins[card]
        return
    points = sum(t.points[x] for _, x in trick)
    if any(wins.values()) and points >= 10:
        c["can_win_valuable_last"] += 1
        c["ducks_valuable_trick"] += not wins[card]
    losing = [x for x in legal if not wins[x]]
    if not wins[card] and losing:
        c["discards_to_opps_last"] += 1
        c["throws_points"] += t.points[card] >= 10 and min(t.points[x] for x in losing) < 10
