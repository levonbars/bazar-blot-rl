"""Combination announcements over a deal: class at hand 1, show at hand 2 by the comparison's
winning team, blot/reblot as the trump K and Q are played (rules §6.2, §6.5)."""

from __future__ import annotations

import itertools
import random

from bazarblot.core.announcements import Announcement, announced_class, announcements
from bazarblot.core.auction import BidAction, PassAction, opener_seat
from bazarblot.core.cards import TEAM_OF, card_id
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.declarations import detect_hand_melds
from bazarblot.core.rules import load_default

RULES = load_default()


def cards(*labels: str) -> frozenset[int]:
    return frozenset(card_id(label[-1], label[:-1]) for label in labels)


def _play_deal(
    hands: tuple[frozenset[int], ...], contract_type: str, rng: random.Random, dealer: int = 0
) -> tuple[Deal, list[list[Announcement]]]:
    """Opener bids 8 of `contract_type`, everyone else passes, then random legal cards. Returns
    the deal and the announcement list after every card."""
    deal = Deal(RULES, dealer=dealer, hands=hands, deal_id=0)
    deal.step(BidAction(8, contract_type))  # type: ignore[arg-type]
    while deal.phase == Phase.AUCTION:
        deal.step(PassAction())
    assert announcements(deal) == []
    timeline = []
    while deal.phase == Phase.PLAY:
        deal.step(rng.choice(deal.legal_actions()))
        timeline.append(announcements(deal))
    return deal, timeline


def test_class_is_the_best_combination() -> None:
    assert announced_class(detect_hand_melds(cards("7S", "8S", "9S"), 0, "H", RULES)) == "tierce"
    assert announced_class(detect_hand_melds(cards("7S", "8S", "9S", "10S"), 0, "H", RULES)) == (
        "fifty"
    )
    run6 = cards("7S", "8S", "9S", "10S", "JS", "QS")
    assert announced_class(detect_hand_melds(run6, 0, "H", RULES)) == "hundred"
    carre_and_run = cards("JC", "JD", "JH", "JS", "7S", "8S", "9S", "10S")
    assert announced_class(detect_hand_melds(carre_and_run, 0, "H", RULES)) == "4x"
    assert announced_class(detect_hand_melds(cards("KH", "QH"), 0, "H", RULES)) is None


def test_blot_then_reblot_when_the_cards_are_played() -> None:
    h0 = cards("KH", "QH", "7C", "8D", "9S", "JC", "10D", "AS")
    rest = sorted(set(range(32)) - h0)
    hands = (h0, frozenset(rest[:8]), frozenset(rest[8:16]), frozenset(rest[16:]))
    deal, _ = _play_deal(hands, "H", random.Random(1), dealer=3)
    calls = [a for a in announcements(deal) if a.kind in ("blot", "reblot")]
    assert [(a.seat, a.kind) for a in calls] == [(0, "blot"), (0, "reblot")]
    assert {a.card for a in calls} == set(cards("KH", "QH"))


def test_no_blot_in_no_trump() -> None:
    h0 = cards("KH", "QH", "7C", "8D", "9S", "JC", "10D", "AS")
    rest = sorted(set(range(32)) - h0)
    hands = (h0, frozenset(rest[:8]), frozenset(rest[8:16]), frozenset(rest[16:]))
    deal, _ = _play_deal(hands, "NT", random.Random(1), dealer=3)
    assert not [a for a in announcements(deal) if a.kind in ("blot", "reblot")]


def test_timing_and_totals_over_random_deals() -> None:
    rng = random.Random(2024)
    for i in range(300):
        contract_type = rng.choice(["C", "D", "H", "S", "NT"])
        deal, timeline = _play_deal(deal_hands(rng, RULES), contract_type, rng, dealer=i % 4)
        # announcements only ever grow: nothing said is later unsaid or reordered
        for before, after in itertools.pairwise(timeline):
            assert after[: len(before)] == before
        final = timeline[-1]
        for a in final:
            if a.kind in ("tierce", "fifty", "hundred", "4x"):
                assert a.trick == 0
                hand_melds = detect_hand_melds(
                    deal.original_hands[a.seat],
                    a.seat,
                    contract_type,
                    RULES,  # type: ignore[arg-type]
                )
                assert announced_class(hand_melds) == a.kind
            elif a.kind == "show":
                assert a.trick == 1
        # each seat announces at most once and shows at most once
        for kind in ("show", "class"):
            seats = [
                a.seat
                for a in final
                if (a.kind == "show") == (kind == "show") and a.kind not in ("blot", "reblot")
            ]
            assert len(seats) == len(set(seats))
        # only one team shows, and what was shown or called is exactly what is scored
        shown_teams = {TEAM_OF[a.seat] for a in final if a.kind == "show"}
        assert len(shown_teams) <= 1
        assert deal.result is not None
        points = [0, 0]
        for a in final:
            if a.kind == "show":
                points[TEAM_OF[a.seat]] += sum(m.value for m in a.melds)
            elif a.kind == "reblot":
                points[TEAM_OF[a.seat]] += RULES.combinations.blot_reblot
        att = deal.result.attackers_team
        assert points[att] == deal.result.combo_attackers
        assert points[1 - att] == deal.result.combo_defenders
        # the trick-1 leader is the opener; their class (if any) is the first thing said
        leader = opener_seat(deal.dealer, RULES)
        assert deal.tricks[0].leader == leader
