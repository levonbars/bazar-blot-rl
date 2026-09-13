"""PIMC (Perfect-Information Monte Carlo): sample `K` worlds consistent with the info set (own
hand fixed, known voids respected), DD-solve the continuation of each candidate card, and play
whichever legal card has the best average declaring-team outcome across the `K` samples.

**Auction decisions — and, above `solve_threshold` cards remaining, PLAY decisions too — are
delegated to a `fallback` agent** (a `HeuristicAgent` by default; composition, not inheritance).
PIMC as the roadmap describes it ("sample K worlds..., DD-solve each, play the argmax action") is
fundamentally a PLAY-phase method — it needs a concrete deal to hand the double-dummy solver, and
a *bid's* value depends on how the rest of the auction unfolds, not on a single perfect-
information subgame. Real PIMC-style bridge bots make the same split (search-driven card play,
convention-driven bidding); rebuilding a bidding search here is out of scope for a first baseline.

**`solve_threshold` (M2.5 item 2) is the honest answer to `solve_from`'s cost, not a workaround
that hides it.** A full 8-trick DD solve is 1-40+ seconds (measured, M2), and PIMC needs one solve
per (sampled world, candidate card) pair — `K x len(legal_cards)` solves for a single decision.
At the very first play-phase decision (8 legal cards) that's `8K` full solves before a single
card is chosen; PIMC-20 there costs many minutes. Once `len(hand) <= solve_threshold` (default 8,
i.e. always solve — set it lower, e.g. 4, to only take over near the endgame), remaining hands
are small enough that solves are cheap and close to exhaustive — this is a standard, legitimate
bot design ("endgame solver + heuristic opening"), not a compromise on correctness: PIMC still
solves exactly whenever it solves at all, it just doesn't pretend to be fast everywhere.

**Cross-candidate-card transposition-table reuse (M2.5 item 3).** Every legal card is evaluated
against the *same* sampled world, and those candidates' subtrees overlap heavily — "play card A,
then optimal continuation" and "play card B, then optimal continuation" often converge onto
identical downstream positions a few plies in. `_play` now constructs one `solver.dd.Solver` per
sampled world and reuses it (hence its transposition table) across every candidate card for that
world, rather than the fresh, empty table each `solve_from` call used to start with. This is safe
for exactly the reason `Solver` itself documents: its transposition-table key is the exact game
position throughout, so reusing it across different starting queries is no different in kind from
reusing it across `solve()`'s own internal recursive calls.

See `tests/test_agents_pimc.py` for how this is verified at a tractable scale, and the M4 status
write-up plus M2.5 in the roadmap for the honest scope of what "PIMC-20 beats heuristic" means.
"""

from __future__ import annotations

import random

import numpy as np
import numpy.typing as npt

from bazarblot.core.cards import TEAM_OF, suit_of
from bazarblot.core.deal import Phase
from bazarblot.core.play import current_winner
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace
from bazarblot.env.infoset import InfoSet
from bazarblot.solver.dd import Hands, Solver, Trick

from .base import Agent
from .heuristic import HeuristicAgent


def _infer_voids(info: InfoSet) -> dict[int, set[str]]:
    """seat -> suits that seat is known to be void in, inferred from public trick history.

    Playing off the led suit proves a void in it UNLESS the card played is a trump in a trump
    contract (ruffing while holding the led suit is sometimes legal — see `PlayRules` — so
    ruffing alone does not prove a void). Every other case of "played a different suit than what
    was led" is an unconditional proof: under a trump contract, playing a second, non-trump,
    off-led-suit card proves void in the led suit; under `NT`, any off-led-suit play does.
    """
    voids: dict[int, set[str]] = {s: set() for s in range(4)}
    contract_type = info.contract.contract_type if info.contract is not None else None
    for trick in (*info.tricks, info.current_trick):
        if not trick:
            continue
        led_suit = suit_of(trick[0][1])
        for seat, card in trick[1:]:
            played_suit = suit_of(card)
            if played_suit == led_suit:
                continue
            if contract_type is not None and contract_type != "NT" and played_suit == contract_type:
                continue  # ruffed — doesn't by itself prove a void in the led suit
            voids[seat].add(led_suit)
    return voids


def _value_of_playing(
    solver: Solver,
    hands: Hands,
    to_act: int,
    trick_so_far: Trick,
    card: int,
    last_hand_bonus: int,
) -> int:
    """Declaring-team raw points if `to_act` plays `card` right now and both sides play
    optimally from there on. Mirrors `solver/dd.py`'s own trick-resolution step (gain folded in
    at the exact point a trick completes) rather than calling into its private internals. Takes
    an already-constructed `Solver` — shared across every candidate card evaluated against the
    same sampled world (M2.5 item 3) — rather than the module-level `solve_from` function, which
    would start a fresh, empty transposition table on every single call."""
    declaring_team = solver.declaring_team
    new_hands = list(hands)
    new_hands[to_act] = new_hands[to_act] - {card}
    hands_after: Hands = (new_hands[0], new_hands[1], new_hands[2], new_hands[3])
    new_trick = (*trick_so_far, (to_act, card))

    if len(new_trick) < 4:
        remainder = solver.solve_from(hands_after, (to_act + 1) % 4, new_trick)
        return remainder.declarer_points

    winner = current_winner(new_trick, solver.tables)
    gain = sum(solver.tables.points[c] for _, c in new_trick)
    is_last_trick = not any(hands_after)
    if is_last_trick:
        gain += last_hand_bonus
        return gain if TEAM_OF[winner] == declaring_team else 0
    remainder = solver.solve_from(hands_after, winner, ())
    gain_for_declaring = gain if TEAM_OF[winner] == declaring_team else 0
    return gain_for_declaring + remainder.declarer_points


class PIMCAgent:
    def __init__(
        self,
        rules: RuleConfig,
        k: int = 20,
        fallback: Agent | None = None,
        rng: random.Random | None = None,
        max_sample_attempts: int = 200,
        solve_threshold: int = 8,
    ) -> None:
        self.rules = rules
        self.k = k
        self.fallback = fallback if fallback is not None else HeuristicAgent(rules)
        self.rng = rng if rng is not None else random.Random()
        self.max_sample_attempts = max_sample_attempts
        self.solve_threshold = solve_threshold

    def act(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        if info.phase == Phase.AUCTION:
            return self.fallback.act(info, space, legal_mask)
        if len(info.hand) > self.solve_threshold:
            # Too many cards left for even a handful of full solves to be affordable — play this
            # decision with the fallback instead of pretending to solve it. See module docstring.
            return self.fallback.act(info, space, legal_mask)
        return self._play(info, space, legal_mask)

    def _sample_world(self, info: InfoSet) -> Hands | None:
        """One full 4-hand assignment consistent with `info`: my own hand fixed, the other three
        seats' still-unseen cards redistributed among them at random, respecting each seat's
        known remaining count and any suit it's known to be void in. `None` if no assignment
        satisfying every void constraint was found within `max_sample_attempts` tries (rare —
        only happens when several tight voids interact) — the caller just skips that sample."""
        seat = info.seat
        others = [s for s in range(4) if s != seat]
        played: set[int] = set()
        for trick in info.tricks:
            played |= {c for _, c in trick}
        played |= {c for _, c in info.current_trick}
        unseen = [c for c in range(32) if c not in info.hand and c not in played]

        target_counts = {s: info.hand_sizes[s] for s in others}
        assert sum(target_counts.values()) == len(unseen)
        voids = _infer_voids(info)

        for _attempt in range(self.max_sample_attempts):
            pool = list(unseen)
            self.rng.shuffle(pool)
            assignment: dict[int, list[int]] = {s: [] for s in others}
            ok = True
            for card in pool:
                candidates = [
                    s
                    for s in others
                    if len(assignment[s]) < target_counts[s] and suit_of(card) not in voids[s]
                ]
                if not candidates:
                    ok = False
                    break
                assignment[self.rng.choice(candidates)].append(card)
            if ok:
                hands = [frozenset[int]()] * 4
                hands[seat] = info.hand
                for s in others:
                    hands[s] = frozenset(assignment[s])
                return (hands[0], hands[1], hands[2], hands[3])
        return None

    def _play(self, info: InfoSet, space: ActionSpace, legal_mask: npt.NDArray[np.bool_]) -> int:
        assert info.contract is not None
        legal_cards = [c for c in range(32) if legal_mask[space.play_index(c)]]
        assert legal_cards, "play phase always has at least one legal card"
        if len(legal_cards) == 1:
            return space.play_index(legal_cards[0])  # nothing to decide — skip solving entirely

        contract_type = info.contract.contract_type
        declaring_team = info.contract.attacking_team
        last_hand_bonus = self.rules.contracts.last_hand_bonus
        totals = dict.fromkeys(legal_cards, 0.0)
        counts = dict.fromkeys(legal_cards, 0)

        for _ in range(self.k):
            world = self._sample_world(info)
            if world is None:
                continue
            # One Solver per sampled world, reused across every candidate card below — see
            # module docstring, M2.5 item 3.
            solver = Solver(contract_type, declaring_team, self.rules)
            for card in legal_cards:
                totals[card] += _value_of_playing(
                    solver, world, info.seat, info.current_trick, card, last_hand_bonus
                )
                counts[card] += 1

        scored = [(totals[c] / counts[c], c) for c in legal_cards if counts[c] > 0]
        if not scored:
            card = self.rng.choice(legal_cards)  # every sample failed — extremely unlikely
        else:
            am_declaring = TEAM_OF[info.seat] == declaring_team
            target = max(scored)[0] if am_declaring else min(scored)[0]
            card = next(c for v, c in scored if v == target)
        return space.play_index(card)
