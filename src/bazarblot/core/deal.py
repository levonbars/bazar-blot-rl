"""One full deal: auction phase, then play phase, then scoring. The engine's top-level API.

`Deal` is a small mutable state machine (turn-based games are naturally sequential; fighting
that with immutable-everything buys little here). The rules logic itself stays in the pure
`auction`/`play`/`declarations`/`scoring` modules — this class only sequences calls into them
and owns the bookkeeping (hands, tricks, whose turn it is).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from bazarblot.core.auction import (
    AuctionAction,
    AuctionState,
    BidAction,
    ContraAction,
    Contract,
    PassAction,
    RecontraAction,
    final_contract,
    new_auction,
    opener_seat,
)
from bazarblot.core.auction import (
    apply as apply_auction,
)
from bazarblot.core.cards import N_CARDS, TEAM_OF, TRICKS_PER_DEAL, build_tables
from bazarblot.core.declarations import Meld, resolve_combinations
from bazarblot.core.play import PlayCardAction, Trick, current_winner, legal_moves
from bazarblot.core.scoring import SaysTeam, compute_card_portions, score_deal

if TYPE_CHECKING:
    from bazarblot.core.cards import ContractTables
    from bazarblot.core.rules import RuleConfig

Action = AuctionAction | PlayCardAction


class Phase(Enum):
    AUCTION = auto()
    PLAY = auto()
    TERMINAL = auto()
    ABORTED = auto()


class DealError(ValueError):
    pass


class IllegalActionError(DealError):
    pass


class DealFinishedError(DealError):
    pass


@dataclass(frozen=True, slots=True)
class DealResult:
    contract: Contract
    attackers_team: int
    cards_attackers: int
    cards_defenders: int
    combo_attackers: int
    combo_defenders: int
    raw_attackers: int
    raw_defenders: int
    made: bool
    score_attackers: int
    score_defenders: int
    attackers_took_all_tricks: bool
    defenders_took_all_tricks: bool
    winning_meld: Meld | None

    @property
    def team_scores(self) -> tuple[int, int]:
        """(team 0 score, team 1 score), regardless of which team attacked."""
        scores = [0, 0]
        scores[self.attackers_team] = self.score_attackers
        scores[1 - self.attackers_team] = self.score_defenders
        return (scores[0], scores[1])


class Deal:
    """Full-information ground truth for one deal. See `InfoSet` (env layer, not M1) for what
    an individual player is actually allowed to see — this class is not that boundary.
    """

    def __init__(
        self,
        rules: RuleConfig,
        dealer: int,
        hands: tuple[frozenset[int], ...],
        deal_id: int = 0,
    ) -> None:
        if len(hands) != 4:
            raise DealError(f"expected 4 hands, got {len(hands)}")
        seen: set[int] = set()
        for h in hands:
            if len(h) != rules.deal.cards_per_player:
                raise DealError(f"hand has {len(h)} cards, expected {rules.deal.cards_per_player}")
            if seen & h:
                raise DealError("a card appears in more than one hand")
            seen |= h
        # No separate "covers all N_CARDS" check: 4 pairwise-disjoint hands of exactly
        # `cards_per_player` each already sum to N_CARDS by construction once the two
        # checks above pass — asserting it again would be dead code, not a real invariant.

        self.rules = rules
        self.dealer = dealer
        self.deal_id = deal_id
        self.original_hands: tuple[frozenset[int], ...] = tuple(frozenset(h) for h in hands)
        self.hands: list[frozenset[int]] = list(self.original_hands)

        self.phase = Phase.AUCTION
        self.auction_state: AuctionState = new_auction(dealer, rules)
        self.contract: Contract | None = None
        self.tables: ContractTables | None = None

        self.tricks: list[Trick] = []
        self.current_trick: list[tuple[int, int]] = []
        self.trick_leader: int | None = None

        self.result: DealResult | None = None

    # ---------------------------------------------------------------- turn order

    @property
    def to_act(self) -> int:
        if self.phase == Phase.AUCTION:
            return self.auction_state.to_act
        if self.phase == Phase.PLAY:
            assert self.trick_leader is not None
            return (self.trick_leader + len(self.current_trick)) % 4
        raise DealFinishedError(f"no one to act: deal is in phase {self.phase}")

    # ---------------------------------------------------------------- legal actions

    def legal_actions(self) -> tuple[Action, ...]:
        if self.phase == Phase.PLAY:
            assert self.contract is not None and self.tables is not None
            hand = self.hands[self.to_act]
            moves = legal_moves(
                hand,
                self.current_trick,
                self.to_act,
                self.contract.contract_type,
                self.rules,
                self.tables,
            )
            return tuple(PlayCardAction(c) for c in moves)
        raise DealFinishedError(
            f"legal_actions() is only supported in PLAY phase (got {self.phase}); "
            "use bazarblot.core.auction.is_legal to check a specific auction action"
        )

    # ---------------------------------------------------------------- stepping

    def step(self, action: Action) -> None:
        if self.phase == Phase.AUCTION:
            if not isinstance(action, PassAction | BidAction | ContraAction | RecontraAction):
                raise IllegalActionError(f"expected an auction action, got {action!r}")
            self._step_auction(action)
        elif self.phase == Phase.PLAY:
            if not isinstance(action, PlayCardAction):
                raise IllegalActionError(f"expected a PlayCardAction, got {action!r}")
            self._step_play(action)
        else:
            raise DealFinishedError(f"deal is finished (phase={self.phase}), cannot step")

    def _step_auction(self, action: AuctionAction) -> None:
        self.auction_state = apply_auction(self.auction_state, action, self.rules)
        if not self.auction_state.finished:
            return
        if self.auction_state.aborted:
            self.phase = Phase.ABORTED
            return
        contract = final_contract(self.auction_state)
        assert contract is not None
        self.contract = contract
        self.tables = build_tables(contract.contract_type, self.rules)
        self.phase = Phase.PLAY
        self.trick_leader = opener_seat(self.dealer, self.rules)

    def _step_play(self, action: PlayCardAction) -> None:
        assert self.contract is not None and self.tables is not None
        seat = self.to_act
        card = action.card
        hand = self.hands[seat]
        if card not in hand:
            raise IllegalActionError(f"seat {seat} does not hold card {card}")
        legal = legal_moves(
            hand, self.current_trick, seat, self.contract.contract_type, self.rules, self.tables
        )
        if card not in legal:
            raise IllegalActionError(f"card {card} is not legal for seat {seat}: legal={legal}")

        self.hands[seat] = hand - {card}
        self.current_trick.append((seat, card))

        if len(self.current_trick) < 4:
            return

        winner = current_winner(self.current_trick, self.tables)
        points = sum(self.tables.points[c] for _, c in self.current_trick)
        assert self.trick_leader is not None
        trick = Trick(
            leader=self.trick_leader, plays=tuple(self.current_trick), winner=winner, points=points
        )
        self.tricks.append(trick)
        self.current_trick = []
        self.trick_leader = winner

        if len(self.tricks) == TRICKS_PER_DEAL:
            self._finish_play()

    # ---------------------------------------------------------------- scoring

    def _finish_play(self) -> None:
        assert self.contract is not None
        contract = self.contract
        cards_a_face = sum(
            t.points for t in self.tricks if TEAM_OF[t.winner] == contract.attacking_team
        )
        cards_d_face = sum(
            t.points for t in self.tricks if TEAM_OF[t.winner] != contract.attacking_team
        )

        attackers_all = all(TEAM_OF[t.winner] == contract.attacking_team for t in self.tricks)
        defenders_all = all(TEAM_OF[t.winner] != contract.attacking_team for t in self.tricks)
        last_winner = self.tricks[-1].winner
        says_team: SaysTeam = (
            "attackers" if TEAM_OF[last_winner] == contract.attacking_team else "defenders"
        )

        cards_a, cards_d = compute_card_portions(
            card_points_attackers=cards_a_face,
            card_points_defenders=cards_d_face,
            says_team=says_team,
            attackers_all_tricks=attackers_all,
            defenders_all_tricks=defenders_all,
            rules=self.rules,
        )

        leader_seat = opener_seat(self.dealer, self.rules)  # trick-1 leader, for elder-hand ties
        combo = resolve_combinations(
            self.original_hands, contract.contract_type, self.rules, leader_seat
        )
        combo_a = combo.team_points[contract.attacking_team]
        combo_d = combo.team_points[1 - contract.attacking_team]

        score = score_deal(
            rules=self.rules,
            contract=contract,
            cards_attackers=cards_a,
            cards_defenders=cards_d,
            combo_attackers=combo_a,
            combo_defenders=combo_d,
            attackers_all_tricks=attackers_all,
            defenders_all_tricks=defenders_all,
        )

        self.result = DealResult(
            contract=contract,
            attackers_team=contract.attacking_team,
            cards_attackers=cards_a,
            cards_defenders=cards_d,
            combo_attackers=combo_a,
            combo_defenders=combo_d,
            raw_attackers=score.raw_attackers,
            raw_defenders=score.raw_defenders,
            made=score.made,
            score_attackers=score.attackers_score,
            score_defenders=score.defenders_score,
            attackers_took_all_tricks=attackers_all,
            defenders_took_all_tricks=defenders_all,
            winning_meld=combo.winning_meld,
        )
        self.phase = Phase.TERMINAL

    # ---------------------------------------------------------------- invariants (§2.3)

    def check_invariants(self) -> None:
        """Card-conservation and structural checks. Cheap; run freely in tests."""
        in_hands = sum(len(h) for h in self.hands)
        in_tricks = 4 * len(self.tricks)
        in_current = len(self.current_trick)
        total = in_hands + in_tricks + in_current
        if total != N_CARDS:
            raise DealError(f"card conservation violated: {total} != {N_CARDS}")

        seen: set[int] = set()
        for h in self.hands:
            seen |= h
        for _, c in self.current_trick:
            seen.add(c)
        for t in self.tricks:
            for _, c in t.plays:
                seen.add(c)
        if len(seen) != total:
            raise DealError("a card appears more than once across hands/tricks")

        if self.phase == Phase.PLAY:
            if not (0 <= len(self.current_trick) < 4):
                raise DealError(f"current_trick has {len(self.current_trick)} plays")
            if self.to_act < 0 or self.to_act > 3:
                raise DealError(f"to_act out of range: {self.to_act}")

        if self.phase == Phase.TERMINAL:
            if len(self.tricks) != TRICKS_PER_DEAL:
                raise DealError(
                    f"terminal with {len(self.tricks)} tricks, expected {TRICKS_PER_DEAL}"
                )
            if self.result is None:
                raise DealError("terminal phase with no result")
