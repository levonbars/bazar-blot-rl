"""M1.5: the information-hiding boundary. `player_view` must never leak a card that is still
in another seat's unplayed hand — the whole point of `play` mode existing as a separate code
path from `watch`/`replay`.
"""

from __future__ import annotations

import random

from bazarblot.core.auction import BidAction, PassAction
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import load_default
from bazarblot.ui.views import PlayerView, full_state_view, player_view
from tests.helpers import random_auction_action

RULES = load_default()


def _card_bearing_ints(view: PlayerView) -> set[int]:
    """Every int that is actually a card identity in a `PlayerView` — deliberately NOT every
    int in the structure. Fields like bid levels, seat numbers, dealer and match scores are
    small public integers in the same 0-31 numeric range as card ids; flattening "every int
    anywhere" produces false-positive "leaks" on pure numeric coincidence (e.g. a bid level
    of 8 colliding with card_id 8), which is a bug in a leakage TEST, not in the view. Only
    fields that are actually card slots count.
    """
    out: set[int] = set(view.my_hand)
    for _seat, card in view.current_trick:
        out.add(card)
    for trick in view.completed_tricks:
        for _seat, card in trick.plays:
            out.add(card)
    for action in view.legal_actions:
        if action.get("type") == "play":
            out.add(action["card"])
    return out


def test_player_view_never_contains_another_seats_unplayed_cards() -> None:
    """The core leakage property, over many partially-played deals at every phase."""
    for seed in range(300):
        rng = random.Random(seed)
        hands = deal_hands(rng, RULES)
        d = Deal(RULES, dealer=seed % 4, hands=hands, deal_id=seed)

        # advance a random number of steps into the auction/play so hands are partially
        # depleted and some cards have genuinely moved from "hidden" to "public" (played)
        n_steps = rng.randint(0, 20)
        for _ in range(n_steps):
            if d.phase == Phase.AUCTION:
                d.step(random_auction_action(d.auction_state, RULES, rng))
            elif d.phase == Phase.PLAY:
                legal = d.legal_actions()
                if not legal:
                    break
                d.step(rng.choice(legal))
            else:
                break

        if d.phase not in (Phase.AUCTION, Phase.PLAY):
            continue  # terminal/aborted deals have no hidden hands left to leak

        for seat in range(4):
            view = player_view(d, seat, (0, 0), RULES.match.target)
            visible_cards = _card_bearing_ints(view)
            other_hidden_cards: set[int] = set()
            for other in range(4):
                if other != seat:
                    other_hidden_cards |= d.hands[other]
            leaked = visible_cards & other_hidden_cards
            assert not leaked, (
                f"seed={seed} seat={seat}: player_view leaked hidden cards {leaked} "
                f"belonging to other seats"
            )
            # and the positive control: my own hand must be exactly right, not just "not leaked"
            assert set(view.my_hand) == d.hands[seat]


def test_player_view_hand_sizes_are_public_but_never_expose_identities() -> None:
    rng = random.Random(42)
    hands = deal_hands(rng, RULES)
    d = Deal(RULES, dealer=0, hands=hands, deal_id=0)
    view = player_view(d, seat=0, match_score=(0, 0), match_target=101)
    assert view.hand_sizes == (8, 8, 8, 8)
    # hand_sizes is the ONLY place other seats' hand information appears at all
    assert not hasattr(view, "hands")


def test_full_state_view_legitimately_contains_all_four_hands() -> None:
    """The contrasting positive case: `watch`/`replay` are SUPPOSED to see everything."""
    rng = random.Random(7)
    hands = deal_hands(rng, RULES)
    d = Deal(RULES, dealer=0, hands=hands, deal_id=0)
    view = full_state_view(d, (0, 0), RULES.match.target)
    for seat in range(4):
        assert set(view.hands[seat]) == d.hands[seat]


def test_legal_actions_only_populated_for_the_viewing_seat() -> None:
    rng = random.Random(1)
    hands = deal_hands(rng, RULES)
    d = Deal(RULES, dealer=0, hands=hands, deal_id=0)
    d.step(PassAction())
    d.step(BidAction(level=8, contract_type="H"))
    to_act = d.auction_state.to_act
    for seat in range(4):
        view = player_view(d, seat, (0, 0), RULES.match.target)
        if seat == to_act:
            assert view.legal_actions != []
        else:
            assert view.legal_actions == []
