"""M1.5: the placeholder bot always produces a legal action. Not a strength/quality test —
see `bots.py`'s own docstring for why this bot is intentionally simple."""

from __future__ import annotations

import random

from bazarblot.core.auction import is_legal
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import load_default
from bazarblot.ui.bots import bot_action

RULES = load_default()


def test_bot_always_acts_legally_through_full_deals() -> None:
    for seed in range(200):
        rng = random.Random(seed)
        hands = deal_hands(rng, RULES)
        d = Deal(RULES, dealer=seed % 4, hands=hands, deal_id=seed)

        while d.phase in (Phase.AUCTION, Phase.PLAY):
            seat = d.to_act
            action = bot_action(d, seat, rng)
            if d.phase == Phase.AUCTION:
                assert is_legal(d.auction_state, action, RULES), (  # type: ignore[arg-type]
                    f"seed={seed}: bot proposed illegal auction action {action!r}"
                )
            else:
                assert action in d.legal_actions(), (
                    f"seed={seed}: bot proposed illegal play action {action!r}"
                )
            d.step(action)
            d.check_invariants()

        assert d.phase in (Phase.TERMINAL, Phase.ABORTED)


def test_bot_never_recontras_and_always_accepts_a_contra() -> None:
    """Documented simplification in bots.py: the bot always accepts a contra rather than
    ever redoubling."""
    from bazarblot.core.auction import BidAction, ContraAction, PassAction, apply, new_auction
    from bazarblot.ui.bots import bot_auction_action

    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)
    state = apply(state, ContraAction(), RULES)
    assert state.doubling == "contra" and not state.finished

    hand = frozenset(range(8))  # contents irrelevant — the bot never recontras regardless
    action = bot_auction_action(state, hand, RULES)
    assert isinstance(action, PassAction)

    result = apply(state, action, RULES)
    assert result.finished
    assert result.doubling == "contra"  # accepted, never upgraded to recontra
