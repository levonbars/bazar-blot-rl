"""M1: the auction state machine. Rules §4."""

from __future__ import annotations

import pytest

from bazarblot.core.auction import (
    AuctionError,
    BidAction,
    ContraAction,
    PassAction,
    RecontraAction,
    apply,
    final_contract,
    is_legal,
    new_auction,
)
from bazarblot.core.rules import load_default

RULES = load_default()


def test_opener_is_dealer_left_and_also_will_lead_trick_1() -> None:
    state = new_auction(dealer=0, rules=RULES)
    assert state.to_act == 1
    state = new_auction(dealer=3, rules=RULES)
    assert state.to_act == 0


def test_minimum_opening_bid() -> None:
    state = new_auction(dealer=0, rules=RULES)
    assert not is_legal(state, BidAction(level=7, contract_type="H"), RULES)
    assert is_legal(state, BidAction(level=8, contract_type="H"), RULES)


def test_maximum_bid_is_eighty() -> None:
    state = new_auction(dealer=0, rules=RULES)
    assert is_legal(state, BidAction(level=80, contract_type="H", capot=True), RULES)
    assert not is_legal(state, BidAction(level=81, contract_type="H", capot=True), RULES)


def test_bid_must_strictly_exceed_standing_level() -> None:
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=12, contract_type="H"), RULES)
    assert not is_legal(state, BidAction(level=12, contract_type="S"), RULES)
    assert is_legal(state, BidAction(level=13, contract_type="S"), RULES)


def test_seniority_is_purely_numeric_nt_does_not_outrank_at_equal_level() -> None:
    """You cannot say "12 boy" over "12 hearts" — only the number matters."""
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=12, contract_type="H"), RULES)
    assert not is_legal(state, BidAction(level=12, contract_type="NT"), RULES)


def test_capot_does_not_outrank_a_plain_bid_at_equal_level() -> None:
    """ "26 Capot" cannot answer a standing plain "26" — must go to 27 Capot."""
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=26, contract_type="S"), RULES)
    assert not is_legal(state, BidAction(level=26, contract_type="S", capot=True), RULES)
    assert is_legal(state, BidAction(level=27, contract_type="S", capot=True), RULES)


def test_capot_is_sticky_upward() -> None:
    """Once anyone bids capot, every higher bid must also be capot."""
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=26, contract_type="S", capot=True), RULES)
    assert not is_legal(state, BidAction(level=27, contract_type="S", capot=False), RULES)
    assert is_legal(state, BidAction(level=27, contract_type="S", capot=True), RULES)


def test_pass_is_not_binding_a_passed_player_may_bid_again() -> None:
    state = new_auction(dealer=0, rules=RULES)  # opener = seat 1
    state = apply(state, PassAction(), RULES)  # seat 1 passes
    state = apply(state, BidAction(level=8, contract_type="H"), RULES)  # seat 2 bids
    state = apply(state, PassAction(), RULES)  # seat 3 passes
    state = apply(state, PassAction(), RULES)  # seat 0 passes
    # back to seat 1, who passed once already — must still be able to bid now
    assert state.to_act == 1
    assert is_legal(state, BidAction(level=9, contract_type="H"), RULES)
    assert not state.finished  # only 2 consecutive passes (seat3, seat0) since the bid, not 3


def test_four_passes_with_no_bid_aborts() -> None:
    state = new_auction(dealer=0, rules=RULES)
    for _ in range(4):
        state = apply(state, PassAction(), RULES)
    assert state.finished
    assert state.aborted
    assert final_contract(state) is None


def test_last_bidder_may_raise_variant_needs_four_passes_not_three() -> None:
    """[OPEN-2c] alternative: after 3 passes the turn naturally cycles back to the bidder
    (who is 1 of 4 seats), giving them a final chance to raise their own contract instead of
    it auto-closing. If they also pass, that's 4 consecutive -> now it closes."""
    import copy

    from bazarblot.core.rules import RuleConfig

    data = copy.deepcopy(RULES.raw)
    data["auction"]["three_passes_after_bid"] = "last_bidder_may_raise"
    variant_rules = RuleConfig.from_dict(data)

    state = new_auction(dealer=0, rules=variant_rules)  # opener = seat 1
    state = apply(state, BidAction(level=10, contract_type="H"), variant_rules)  # seat 1 bids
    state = apply(state, PassAction(), variant_rules)  # seat 2
    state = apply(state, PassAction(), variant_rules)  # seat 3
    assert not state.finished  # 2 passes so far; "close" would still be open too
    state = apply(state, PassAction(), variant_rules)  # seat 0 — 3rd consecutive pass
    assert not state.finished  # differs from "close": still open, back to the bidder
    assert state.to_act == 1  # seat 1, the original bidder, gets the final say

    # the bidder may now raise instead of letting it close
    assert is_legal(state, BidAction(level=11, contract_type="H"), variant_rules)
    raised = apply(state, BidAction(level=11, contract_type="H"), variant_rules)
    assert not raised.finished
    assert raised.consecutive_passes == 0  # the cycle restarts


def test_last_bidder_may_raise_variant_closes_on_the_fourth_pass() -> None:
    import copy

    from bazarblot.core.rules import RuleConfig

    data = copy.deepcopy(RULES.raw)
    data["auction"]["three_passes_after_bid"] = "last_bidder_may_raise"
    variant_rules = RuleConfig.from_dict(data)

    state = new_auction(dealer=0, rules=variant_rules)
    state = apply(state, BidAction(level=10, contract_type="H"), variant_rules)  # seat 1
    for _ in range(4):  # seats 2, 3, 0, then back to 1 (the bidder) — all pass
        state = apply(state, PassAction(), variant_rules)
    assert state.finished
    contract = final_contract(state)
    assert contract is not None
    assert contract.level == 10
    assert contract.declarer_seat == 1


def test_three_passes_after_a_bid_closes_the_auction() -> None:
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)  # seat 1 bids
    for _ in range(3):
        state = apply(state, PassAction(), RULES)  # seats 2, 3, 0 pass
    assert state.finished
    assert not state.aborted
    contract = final_contract(state)
    assert contract is not None
    assert contract.level == 10
    assert contract.declarer_seat == 1
    assert contract.attacking_team == 1  # TEAM_OF[1]


def test_contra_only_available_to_the_defending_team() -> None:
    state = new_auction(dealer=0, rules=RULES)  # opener seat 1
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)  # seat 1 (team 1) bids
    # seat 2 is team 0 (defending) — contra legal
    assert is_legal(state, ContraAction(), RULES)
    state2 = apply(state, PassAction(), RULES)
    # seat 3 is team 1 (attacking team itself) — contra illegal
    assert not is_legal(state2, ContraAction(), RULES)


def test_contra_opens_a_reply_window_it_does_not_finish_the_auction_alone() -> None:
    """Corrects an earlier bug: contra used to set finished=True immediately, which made
    recontra structurally unreachable through real play. §4.3."""
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)  # seat 1 bids
    state = apply(state, ContraAction(), RULES)  # seat 2 contras
    assert not state.finished
    assert state.doubling == "contra"
    assert state.to_act == 1  # turn passes specifically to the declarer, not the next seat


def test_only_the_declarer_may_reply_to_a_contra() -> None:
    """Not the declarer's partner, not the defenders — the specific seat who bid."""
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)  # seat 1
    state = apply(state, ContraAction(), RULES)  # seat 2
    assert state.to_act == 1
    assert not is_legal(state, BidAction(level=11, contract_type="H"), RULES)
    assert not is_legal(state, ContraAction(), RULES)
    assert is_legal(state, PassAction(), RULES)
    assert is_legal(state, RecontraAction(), RULES)


def test_declarer_accepts_the_contra_by_passing() -> None:
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)
    state = apply(state, ContraAction(), RULES)
    state = apply(state, PassAction(), RULES)  # accept
    assert state.finished
    assert state.doubling == "contra"
    contract = final_contract(state)
    assert contract is not None
    assert contract.doubling == "contra"


def test_declarer_recontras_in_reply() -> None:
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)
    state = apply(state, ContraAction(), RULES)
    state = apply(state, RecontraAction(), RULES)
    assert state.finished
    assert state.doubling == "recontra"
    contract = final_contract(state)
    assert contract is not None
    assert contract.doubling == "recontra"


def test_recontra_illegal_before_any_contra() -> None:
    state = new_auction(dealer=0, rules=RULES)
    state = apply(state, BidAction(level=10, contract_type="H"), RULES)
    assert not is_legal(state, RecontraAction(), RULES)


def test_bid_multiplier_table_matches_doubling_state() -> None:
    """Full life cycle: none -> contra -> the multiplier used for scoring should reflect it."""
    from bazarblot.core.rules import load_default as _ld

    r = _ld()
    assert r.scoring.multiplier("H", "none") == 1
    assert r.scoring.multiplier("H", "contra") == 2
    assert r.scoring.multiplier("H", "recontra") == 4
    assert r.scoring.multiplier("NT", "none") == 2
    assert r.scoring.multiplier("NT", "contra") == 3
    assert r.scoring.multiplier("NT", "recontra") == 5


def test_is_legal_false_once_finished() -> None:
    state = new_auction(dealer=0, rules=RULES)
    for _ in range(4):
        state = apply(state, PassAction(), RULES)
    assert state.finished
    assert not is_legal(state, PassAction(), RULES)
    assert not is_legal(state, BidAction(level=8, contract_type="H"), RULES)


def test_is_legal_rejects_a_contract_type_outside_this_rules_config() -> None:
    import copy

    from bazarblot.core.rules import RuleConfig

    data = copy.deepcopy(RULES.raw)
    data["contracts"]["types"] = ["H", "S"]  # C, D and NT no longer offered
    restricted = RuleConfig.from_dict(data)
    state = new_auction(dealer=0, rules=restricted)
    assert not is_legal(state, BidAction(level=8, contract_type="C"), restricted)
    assert is_legal(state, BidAction(level=8, contract_type="H"), restricted)


def test_contra_illegal_without_a_standing_bid() -> None:
    state = new_auction(dealer=0, rules=RULES)
    assert not is_legal(state, ContraAction(), RULES)


def test_illegal_action_raises() -> None:
    state = new_auction(dealer=0, rules=RULES)
    with pytest.raises(AuctionError):
        apply(state, BidAction(level=7, contract_type="H"), RULES)  # below min_bid


def test_max_auction_steps_safety_net() -> None:
    """Passing is non-binding, so a policy that only ever passes must eventually hit the
    4-pass abort long before any safety net — verify the safety net exists and fires when
    genuinely exceeded (constructed via a tiny max_auction_steps override)."""
    import copy

    from bazarblot.core.rules import RuleConfig

    data = copy.deepcopy(RULES.raw)
    data["auction"]["max_auction_steps"] = 1
    tiny_rules = RuleConfig.from_dict(data)
    state = new_auction(dealer=0, rules=tiny_rules)
    state = apply(state, PassAction(), tiny_rules)  # step 1, ok
    with pytest.raises(AuctionError):
        apply(state, PassAction(), tiny_rules)  # step 2, exceeds cap of 1
