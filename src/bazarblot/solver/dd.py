"""Double-dummy solver: given all four hands, a contract type, and a leader, compute the
maximum RAW CARD POINTS the declaring team can force under optimal play from both sides.

"Double-dummy" means both partnerships play with full information (all 32 cards visible) and
perfect coordination — the standard perfect-information oracle used to evaluate real (blind)
play, generate bidding ground truth, and distill into a learned value function. It is NOT a
model of how the game is actually played (players don't see each other's hands); it's a ceiling.

This module deliberately reuses `core.play.legal_moves` / `current_winner` and
`core.cards.build_tables` rather than reimplementing trick legality or resolution. Two reasons:
it can't silently diverge from the rules engine's own (extensively tested) behaviour, and it
means the brute-force/alpha-beta agreement tests below are validating the SEARCH algorithm, not
re-deriving whether `core/` implements the rules correctly — that's already M1's job.

**The last-hand bonus is part of the value being searched, not a post-hoc addition.** A rational
line may trade a raw point to secure the 10-point bonus for winning trick 8. If the bonus were
added after the fact — to whatever principal variation the (bonus-unaware) search happened to
settle on — the search would never discover such a trade. `_trick_gain` folds the bonus in at
the exact point a trick empties every hand, so it is part of what gets maximized/minimized from
the start, in both the fast solver and the independent `brute_force_solve` reference.

**On equivalence classes: four independent soundness bugs were found chasing this, all worth
knowing before touching this code again.** The classic double-dummy speedup from bridge solvers
collapses rank-adjacent, still-outstanding cards of a suit into one equivalence class to cut the
branching factor and let a transposition table hit across positions that only differ by which
member of a class is where. Four distinct problems showed up trying it here:

1. **Not sound in the naive (rank-adjacency-only) form**, because this is a POINT-scoring game,
   not a trick-counting one. Strength-adjacent cards routinely have different point values here
   (trump: J=20 sits directly above 9=14), so playing the higher vs. the lower changes how many
   points a trick is worth even though both "beat" the same other cards. Fix: only ever group
   same-suit, same-VALUE, rank-adjacent cards. That grouping IS point-safe.
2. **A `frozenset`-based canonical hand loses multiplicity.** Whenever a player holds more than
   one card from the same equivalence class (e.g. both the 7 and 8 of a suit, both worth 0 under
   most contracts), a `frozenset` of canonical ids collapses the two into one entry, so two
   genuinely different real hands can collide on the same cache key.
3. **Collapsing cards across DIFFERENT hands into a shared canonical id is unsound even with
   (1) and (2) both fixed**, because it can erase which specific hand holds the higher vs. lower
   card of a pair — and that identity matters whenever both members of the pair end up opposing
   each other in the same trick: whoever holds the strictly stronger of the two wins that
   head-to-head, and if the two holders are on opposite teams that changes who wins the trick
   (and therefore who leads next), even though the canonical representation of both positions is
   identical. This was caught empirically: `reconstruct_pv` reached a position with exactly one
   legal move whose value did not match the value the (cross-hand-reduced) search had cached for
   it, which is only possible if two genuinely different real positions shared a key.
4. **Even restricted to "only dedupe a single mover's own equivalent cards" (never touching the
   TT key at all), pruning is still unsound if a THIRD card from the same class sits in another
   hand.** The naive argument was: two cards X, Y held by the same mover, same suit/value,
   nothing between them in strength — play either now, the immediate trick is identical, and the
   one left behind is "equally equivalent" for later. That inductive step is false the moment a
   class-mate Z is held by someone else. Whichever of X/Y the mover keeps determines which one
   Z eventually meets head-to-head in some later trick (say the suit gets led again once no
   third card separates them) — X beats Z, Y doesn't (or vice versa), so if Z's holder is on the
   opposing team, keeping X vs. keeping Y changes who wins that later trick. Caught by the
   brute-force cross-check on a 3-card-per-player reduced deal (seed 1661): a class of three
   same-value clubs split 1-2 across two opposing hands, pruning the mover's pair down to one
   representative, changed the searched value from 13 to 3.

Given four independent bugs, three of them fatal to any cross-hand form, this module keeps the
transposition-table key fully exact — hands stay keyed on real card ids throughout, never a
canonical id — and restricts equivalence reasoning to one narrow, verified-safe case: the mover
may skip a redundant sibling move only when it holds **every** outstanding member of that
same-suit, same-value, strength-contiguous class. When the whole class is in one hand, there is
no third party left to have a head-to-head with, so bug 4's failure mode cannot arise, and the
original "same trick outcome, equally-equivalent card left behind" argument goes through cleanly.
This is a much narrower — and much more modest — speedup than cross-position reduction would
have given, but it is the only form of the four tried here that survived the brute-force
cross-check at every tested scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bazarblot.core.cards import TEAM_OF, ContractTables, ContractType, build_tables, suit_of
from bazarblot.core.play import current_winner, legal_moves

if TYPE_CHECKING:
    from bazarblot.core.rules import RuleConfig

Hands = tuple[frozenset[int], frozenset[int], frozenset[int], frozenset[int]]
Trick = tuple[tuple[int, int], ...]

_EXACT, _LOWER, _UPPER = 0, 1, 2
_BIG = 10**9


@dataclass(frozen=True, slots=True)
class DDResult:
    declarer_points: int
    """Raw card points the declaring team takes under optimal play: face values of cards won,
    PLUS the 10-point last-hand bonus if they win trick 8 (folded into the search, not added
    afterward — see the module docstring). Does NOT include combination bonuses
    (`declarations.py` is a separate, non-search computation) or apply the scoring formula
    (`scoring.py`) — this is the play-phase oracle those consume, not a final deal score."""
    declarer_took_all_tricks: bool
    principal_variation: tuple[int, ...]
    nodes: int


def _remove(hands: Hands, seat: int, card: int) -> Hands:
    new = list(hands)
    new[seat] = new[seat] - {card}
    return (new[0], new[1], new[2], new[3])


def _is_last_trick(hands: Hands) -> bool:
    return not (hands[0] or hands[1] or hands[2] or hands[3])


def _trick_gain(
    new_trick: Trick,
    tables: ContractTables,
    declaring_team: int,
    hands_after: Hands,
    last_hand_bonus: int,
) -> tuple[int, int]:
    """Points this resolved trick contributes to the declaring team's value, and the winner
    seat. The last-hand bonus is included here — at the exact point of resolution — precisely
    so it participates in the search's own maximization rather than being bolted on after."""
    winner = current_winner(new_trick, tables)
    trick_points = sum(tables.points[c] for _, c in new_trick)
    if _is_last_trick(hands_after):
        trick_points += last_hand_bonus
    gain = trick_points if TEAM_OF[winner] == declaring_team else 0
    return gain, winner


def _order_key(tables: ContractTables, card: int) -> int:
    # Try the strongest legal card first — a cheap, commonly-effective default that tends to
    # resolve tricks (and hence trigger cutoffs) earlier in the search.
    return -tables.strength[card]


def _deduped_moves(
    legal: tuple[int, ...], hands: Hands, trick: Trick, tables: ContractTables
) -> list[int]:
    """Legal moves ordered strongest-first, with redundant same-hand equivalent siblings
    dropped (see the module docstring's safe move-pruning argument) — but ONLY when every other
    outstanding card in that same-suit, same-value, strength-contiguous run is ALSO in the
    mover's own hand.

    That last condition is load-bearing, and its absence was a real bug caught by the
    brute-force cross-check: if a class member sits in ANOTHER hand, which of the mover's two
    "equivalent" cards it plays now determines which one is left behind — and if that other
    hand's card later opposes the leftover in a head-to-head (both from the same suit, in the
    same trick), the higher of the two wins it. Since the two are only equal in POINT VALUE, not
    in strength, that head-to-head has a real winner, and if the two holders are on opposite
    teams, playing one now vs. the other changes who wins that later trick. Two same-value
    adjacent cards are genuinely interchangeable only when the mover holds the whole run — no
    outside card can ever contest strength within it, because there's nothing left to contest
    with once every member is already accounted for by the same hand.

    Restricted to the suit(s) actually present in `legal` rather than scanning all 32 cards:
    dedup can only ever collapse cards within one suit, so grouping the full outstanding set on
    every node — most of it irrelevant — was pure overhead. `legal` is usually a single suit
    (follow-suit is the common case); only a free discard while void spans more than one."""
    if len(legal) <= 1:
        return list(legal)

    by_suit: dict[str, list[int]] = {}
    for c in legal:
        by_suit.setdefault(suit_of(c), []).append(c)

    outstanding: frozenset[int] | None = None
    result: list[int] = []
    for suit, cards_in_suit in by_suit.items():
        if len(cards_in_suit) == 1:
            result.append(cards_in_suit[0])
            continue
        if outstanding is None:
            outstanding = hands[0] | hands[1] | hands[2] | hands[3] | {c for _, c in trick}
        # `legal` (hence `cards_in_suit`) is always a subset of the mover's own hand — whichever
        # of the four hands `legal` was generated from. Find it by inclusion.
        mover_hand = next(h for h in hands if set(cards_in_suit) <= h)
        outstanding_suit = sorted(
            (c for c in outstanding if suit_of(c) == suit), key=lambda c: -tables.strength[c]
        )
        group_members: dict[int, list[int]] = {}
        gid = 0
        i = 0
        while i < len(outstanding_suit):
            j = i
            while (
                j + 1 < len(outstanding_suit)
                and tables.points[outstanding_suit[j + 1]] == tables.points[outstanding_suit[i]]
            ):
                j += 1
            group_members[gid] = outstanding_suit[i : j + 1]
            gid += 1
            i = j + 1
        card_to_group = {c: g for g, members in group_members.items() for c in members}

        seen: set[int] = set()
        for card in sorted(cards_in_suit, key=lambda c: -tables.strength[c]):
            g = card_to_group[card]
            fully_owned = all(m in mover_hand for m in group_members[g])
            if fully_owned:
                if g in seen:
                    continue
                seen.add(g)
            result.append(card)

    result.sort(key=lambda c: _order_key(tables, c))
    return result


class _Solver:
    """Single-pass alpha-beta with a transposition table keyed on the EXACT (unreduced)
    position — `(hands, to_act, trick)` — plus safe same-hand equivalent-move pruning. See the
    module docstring for why the TT key stays exact rather than equivalence-reduced."""

    def __init__(
        self,
        tables: ContractTables,
        contract_type: ContractType,
        declaring_team: int,
        rules: RuleConfig,
    ) -> None:
        self.tables = tables
        self.contract_type = contract_type
        self.declaring_team = declaring_team
        self.rules = rules
        self.last_hand_bonus = rules.contracts.last_hand_bonus
        self.tt: dict[tuple[Hands, int, Trick], tuple[int, int, tuple[int, ...]]] = {}
        self.nodes = 0

    def search(
        self, hands: Hands, to_act: int, trick: Trick, alpha: int, beta: int
    ) -> tuple[int, tuple[int, ...]]:
        self.nodes += 1

        if not trick and _is_last_trick(hands):
            return 0, ()

        key = (hands, to_act, trick)
        cached = self.tt.get(key)
        a, b = alpha, beta
        if cached is not None:
            val, flag, pv = cached
            if flag == _EXACT:
                return val, pv
            if flag == _LOWER:
                a = max(a, val)
            elif flag == _UPPER:
                b = min(b, val)
            if a >= b:
                return val, pv

        legal = legal_moves(
            hands[to_act], trick, to_act, self.contract_type, self.rules, self.tables
        )
        moves = _deduped_moves(legal, hands, trick, self.tables)

        maximizing = TEAM_OF[to_act] == self.declaring_team
        best_value = -1 if maximizing else _BIG
        best_pv: tuple[int, ...] = ()

        for card in moves:
            new_hands = _remove(hands, to_act, card)
            new_trick = (*trick, (to_act, card))

            if len(new_trick) == 4:
                gain, winner = _trick_gain(
                    new_trick, self.tables, self.declaring_team, new_hands, self.last_hand_bonus
                )
                child_value, child_pv = self.search(new_hands, winner, (), a - gain, b - gain)
                total = gain + child_value
            else:
                child_value, child_pv = self.search(new_hands, (to_act + 1) % 4, new_trick, a, b)
                total = child_value

            if maximizing:
                if total > best_value:
                    best_value, best_pv = total, (card, *child_pv)
                a = max(a, best_value)
            else:
                if total < best_value:
                    best_value, best_pv = total, (card, *child_pv)
                b = min(b, best_value)

            if a >= b:
                break

        if best_value <= alpha:
            flag = _UPPER
        elif best_value >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self.tt[key] = (best_value, flag, best_pv)
        return best_value, best_pv


class Solver:
    """A reusable double-dummy solver for one `(contract_type, declaring_team, rules)` triple —
    a thin public wrapper around `_Solver` that exposes its transposition table across MULTIPLE
    `solve_from`-style queries, rather than throwing it away after one (M2.5 item 3).

    The motivating caller is `agents/pimc.py`: PIMC evaluates every legal candidate card from the
    same sampled world, and those candidates' subtrees overlap heavily (the last several plies of
    "play this card, then optimal continuation" are often identical or near-identical across
    different first moves). Reusing one `Solver` — hence one transposition table — across that
    whole batch of queries turns repeated, wasted sub-search into cache hits, with no change to
    the algorithm or its correctness: the TT key is the exact `(hands, to_act, trick)` position
    throughout (see the module docstring), so reusing it across different STARTING queries is
    exactly what a persistent transposition table is for — no different, in kind, from reusing it
    across the recursive calls a single `solve()` already makes internally.

    `solve()`/`solve_from()` below are one-shot convenience wrappers around a fresh `Solver` for
    a single query; construct a `Solver` directly and call `solve_from` on it repeatedly whenever
    you have more than one query against the same contract/declaring-team/rules."""

    def __init__(self, contract_type: ContractType, declaring_team: int, rules: RuleConfig) -> None:
        self.contract_type = contract_type
        self.declaring_team = declaring_team
        self.tables = build_tables(contract_type, rules)
        self._solver = _Solver(self.tables, contract_type, declaring_team, rules)

    @property
    def nodes(self) -> int:
        """Cumulative node count across every `solve_from` call made on this instance so far —
        not reset between calls, since the point is to see the whole batch's cost."""
        return self._solver.nodes

    def solve_from(self, hands: Hands, to_act: int, trick_so_far: Trick = ()) -> DDResult:
        """Same contract as the module-level `solve_from` function, minus the arguments this
        instance already fixed at construction (`contract_type`, `declaring_team`, `rules`)."""
        value, pv = self._solver.search(hands, to_act, trick_so_far, 0, _BIG)
        all_tricks = _pv_all_tricks_to_declarer(
            pv, to_act, trick_so_far, self.tables, self.declaring_team
        )
        return DDResult(
            declarer_points=value,
            declarer_took_all_tricks=all_tricks,
            principal_variation=pv,
            nodes=self._solver.nodes,
        )


def solve(
    hands: Hands,
    contract_type: ContractType,
    leader: int,
    declaring_team: int,
    rules: RuleConfig,
) -> DDResult:
    """Solve one deal's play phase to completion, from the start of a fresh trick led by
    `leader`. See `DDResult` for exactly what `declarer_points` does and does not include.
    `solve_from` is the general form this delegates to (`trick_so_far=()`). A one-shot
    convenience wrapper around `Solver` — construct a `Solver` directly for repeated queries."""
    return Solver(contract_type, declaring_team, rules).solve_from(hands, leader, ())


def solve_from(
    hands: Hands,
    contract_type: ContractType,
    to_act: int,
    trick_so_far: Trick,
    declaring_team: int,
    rules: RuleConfig,
) -> DDResult:
    """Like `solve`, but starting from an arbitrary point mid-deal rather than only a fresh
    trick: `trick_so_far` (possibly empty) is what the seats before `to_act`, in this trick's
    play order, have already played into the CURRENT trick, and `hands` holds only the cards
    still unplayed by anyone (nothing in `trick_so_far` may also appear in `hands`).

    Needed by anything that must evaluate "what happens if I play this specific card right now"
    for a seat that is not leading — `_Solver.search` already supports this internally (it's
    exactly what the recursive case does at every non-trick-ending ply), this just exposes it as
    a public entry point. `agents/pimc.py` is the first caller: PIMC has to compare candidate
    cards from whatever position the human/agent seat actually faces, most of which are mid-trick.

    A one-shot convenience wrapper around `Solver` — construct a `Solver` directly and reuse it
    when making more than one query against the same `(contract_type, declaring_team, rules)`."""
    return Solver(contract_type, declaring_team, rules).solve_from(hands, to_act, trick_so_far)


def _pv_all_tricks_to_declarer(
    pv: tuple[int, ...],
    to_act: int,
    initial_trick: Trick,
    tables: ContractTables,
    declaring_team: int,
) -> bool:
    """Whether every trick completed along `pv` (starting from `initial_trick`, possibly
    non-empty for `solve_from`) was won by the declaring team. Purely a replay/reporting
    convenience — has no bearing on `declarer_points`, which already reflects optimal play
    regardless of who wins which trick."""
    trick: list[tuple[int, int]] = list(initial_trick)
    seat = to_act
    for card in pv:
        trick.append((seat, card))
        if len(trick) == 4:
            winner = current_winner(tuple(trick), tables)
            if TEAM_OF[winner] != declaring_team:
                return False
            seat = winner
            trick = []
        else:
            seat = (seat + 1) % 4
    return True


# ---------------------------------------------------------------- brute-force reference


def brute_force_solve(
    hands: Hands, contract_type: ContractType, leader: int, declaring_team: int, rules: RuleConfig
) -> int:
    """Exhaustive minimax, no alpha-beta pruning, no move deduplication, memoized only on the
    exact position. Deliberately independent of `_Solver`'s pruning/ordering machinery, and
    derives the last-hand bonus from its OWN optimal line (see `rec`'s return shape) rather than
    from `solve()` — if it borrowed the fast solver's principal variation for the bonus, a bonus
    bug shared by both would go undetected by the very check meant to catch it. This is the
    reference `solve()` is checked against, not a performance-oriented path: only tractable for
    small deals (a handful of tricks) — see the tests for how far it's pushed.
    """
    tables = build_tables(contract_type, rules)
    last_hand_bonus = rules.contracts.last_hand_bonus
    memo: dict[tuple[Hands, int, Trick], tuple[int, int]] = {}

    def rec(hands: Hands, to_act: int, trick: Trick) -> tuple[int, int]:
        """Returns (value, team_winning_the_final_trick) under optimal play from here."""
        key = (hands, to_act, trick)
        if key in memo:
            return memo[key]

        legal = legal_moves(hands[to_act], trick, to_act, contract_type, rules, tables)
        maximizing = TEAM_OF[to_act] == declaring_team
        best_value = -1 if maximizing else _BIG
        best_final_winner = -1

        for card in legal:
            new_hands = _remove(hands, to_act, card)
            new_trick = (*trick, (to_act, card))
            if len(new_trick) == 4:
                gain, winner = _trick_gain(
                    new_trick, tables, declaring_team, new_hands, last_hand_bonus
                )
                if _is_last_trick(new_hands):
                    total, final_winner = gain, TEAM_OF[winner]
                else:
                    future, final_winner = rec(new_hands, winner, ())
                    total = gain + future
            else:
                total, final_winner = rec(new_hands, (to_act + 1) % 4, new_trick)

            if maximizing:
                if total > best_value:
                    best_value, best_final_winner = total, final_winner
            elif total < best_value:
                best_value, best_final_winner = total, final_winner

        memo[key] = (best_value, best_final_winner)
        return best_value, best_final_winner

    value, _final_winner = rec(hands, leader, ())
    return value
