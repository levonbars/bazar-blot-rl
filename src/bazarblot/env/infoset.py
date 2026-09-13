"""`InfoSet`: what one seat is legally allowed to know, and the single function that derives it
from ground truth. This is the security boundary the whole environment layer sits behind —
`env/actions.py` and `env/obs.py` must only ever receive an `InfoSet`, never a `Deal` or a
`TrackedDeal`. See `tests/test_env_infoset.py`'s leakage test, which is the load-bearing check
for the entire project's results: if an `InfoSet` ever leaks a card id from another seat's
unplayed hand, everything downstream (bid quality, win rate, the paper's numbers) is contaminated
by the agent reading information it could not actually have had.

Mirrors `bazarblot.ui.views.player_view` in spirit (same problem: derive a seat-scoped view of a
full-information state), but is a separate, independent implementation — `env/` depends on
`core/` only, never on `ui/`, and the two boundaries are verified by two separate test suites
rather than one shared implementation both sides trust blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

from bazarblot.core.auction import AuctionState, Contract
from bazarblot.core.cards import TEAM_OF, ContractTables
from bazarblot.core.deal import DealResult, Phase
from bazarblot.core.declarations import Meld, detect_all_melds
from bazarblot.core.play import current_winner
from bazarblot.core.rules import RuleConfig
from bazarblot.env.tracked_deal import AuctionEvent, TrackedDeal


@dataclass(frozen=True, slots=True)
class InfoSet:
    """Everything seat `seat` may legally condition a decision on. Every field here is either
    that seat's own private information (`hand`) or public information every seat can see.

    `melds_by_seat`, when set, is full disclosure of every meld every seat holds — correct ONLY
    under the current `staging.declarations_are_actions = False` default, where "auto-announced
    and auto-shown" means every combination is genuinely visible to everyone from the start of
    play (rules §6.5's staged bluffing sub-game is switched off entirely, not just hidden from
    the observation). `info_set()` raises if that flag is ever turned on, rather than silently
    keeping this field's current (wrong, over-disclosing) semantics — `core/` has no staged
    announce/question/show state machine yet for it to read instead. See `info_set()`.
    """

    seat: int
    rules: RuleConfig
    dealer: int
    deal_number: int
    match_score: tuple[int, int]  # (team 0, team 1), absolute — not seat-relative; obs.py rotates
    phase: Phase
    hand: frozenset[int]  # MY unplayed cards only
    original_hand: frozenset[int]  # MY full 8-card deal — safe, it's my own hand; needed
    # because combinations (§6.1) are fixed by what I was dealt, not by what I still hold
    hand_sizes: tuple[int, int, int, int]  # public: cards remaining per seat
    auction_log: tuple[AuctionEvent, ...]
    auction_state: AuctionState
    contract: Contract | None
    tables: ContractTables | None
    tricks: tuple[tuple[tuple[int, int], ...], ...]  # completed tricks, each a play sequence
    current_trick: tuple[tuple[int, int], ...]
    trick_leader: int | None
    melds_by_seat: tuple[tuple[Meld, ...], ...] | None
    result: DealResult | None

    @property
    def rules_hash(self) -> str:
        return self.rules.rules_hash

    @property
    def to_act(self) -> int:
        if self.phase == Phase.AUCTION:
            return self.auction_state.to_act
        if self.phase == Phase.PLAY:
            assert self.trick_leader is not None
            return (self.trick_leader + len(self.current_trick)) % 4
        raise ValueError(f"no one to act: phase is {self.phase}")

    @property
    def running_points(self) -> tuple[int, int]:
        """Raw card points taken so far this deal, by team (0, 1) — derived from the public
        `tricks` log, not stored redundantly."""
        pts = [0, 0]
        if self.tables is None:
            return (0, 0)
        for trick in self.tricks:
            winner = current_winner(trick, self.tables)
            trick_points = sum(self.tables.points[c] for _, c in trick)
            pts[TEAM_OF[winner]] += trick_points
        return (pts[0], pts[1])


def info_set(
    tracked: TrackedDeal, seat: int, match_score: tuple[int, int], deal_number: int
) -> InfoSet:
    """The ONLY function that reads `TrackedDeal`/`Deal` internals to build an `InfoSet`.
    `match_score` and `deal_number` are supplied by the caller (the AEC/single-agent env, which
    owns the `Match` this deal belongs to) since `Deal` itself has no match context — see
    `env/aec.py`."""
    deal = tracked.deal
    rules = deal.rules

    melds_by_seat: tuple[tuple[Meld, ...], ...] | None = None
    if deal.contract is not None:
        if rules.staging.declarations_are_actions:
            raise NotImplementedError(
                "info_set() only supports staging.declarations_are_actions=False (the auto-"
                "announce/auto-show default). core/ has no staged announce/question/show state "
                "machine for info_set() to read from instead — see declarations.py's docstring."
            )
        # Auto-show means genuinely full disclosure from the start of play, not just at scoring
        # (Deal itself only computes this lazily at _finish_play, for scoring purposes) — so
        # info_set exposes it for every play-phase InfoSet, seat 0's view identical to seat 3's.
        all_melds = detect_all_melds(deal.original_hands, deal.contract.contract_type, rules)
        melds_by_seat = tuple(tuple(m for m in all_melds if m.owner_seat == s) for s in range(4))

    return InfoSet(
        seat=seat,
        rules=rules,
        dealer=deal.dealer,
        deal_number=deal_number,
        match_score=match_score,
        phase=deal.phase,
        hand=deal.hands[seat],
        original_hand=deal.original_hands[seat],
        hand_sizes=(len(deal.hands[0]), len(deal.hands[1]), len(deal.hands[2]), len(deal.hands[3])),
        auction_log=tuple(tracked.auction_log),
        auction_state=deal.auction_state,
        contract=deal.contract,
        tables=deal.tables,
        tricks=tuple(t.plays for t in deal.tricks),
        current_trick=tuple(deal.current_trick),
        trick_leader=deal.trick_leader,
        melds_by_seat=melds_by_seat,
        result=deal.result,
    )
