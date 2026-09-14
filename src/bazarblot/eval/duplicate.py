"""Paired/duplicate deal evaluation (M5, environment spec §9) — the ONLY evaluation entry point
in this project. Belote's per-deal variance is severe: two identical policies can differ by many
points per 100 deals from deal luck alone (a deal with a huge combination sitting in one hand is
just a bigger deal, regardless of who plays it). Duplicate evaluation cancels almost all of that:
play the SAME shuffle twice, with the two policies' seat assignments swapped, and report the
paired difference — standard practice in duplicate bridge, and the only way a modest number of
deals gives a trustworthy signal about which policy is actually better.

**Why aborted deals are resampled at the PAIR level, not redealt within one run.** `env/aec.py`
(live play) transparently redeals a 4-pass abort with a bumped `redeal_attempt` and keeps going —
correct for actually playing a match, where the game must continue. That behavior is wrong here:
if policy X's bidding causes an abort on some shuffle but policy Y's doesn't (or vice versa), and
each run independently redeals-and-retries, the two runs of a "duplicate" pair would silently end
up seeing DIFFERENT hands after that point — exactly the deal-variance contamination duplicate
evaluation exists to remove. `play_deal_once` therefore never retries: it returns `None` on an
abort, and `play_duplicate_pair`/`play_duplicate_match` discard the WHOLE pair and resample a
fresh top-level seed, keeping the "identical shuffle, seats swapped" guarantee exact rather than
approximate. Aborts are rare in practice (a sensible bidder — or even `RandomAgent`, since PASS is
just one of hundreds of legal actions at an opening decision — opens far more often than it
passes four times running), so this costs little.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bazarblot.agents.base import Agent
from bazarblot.core.cards import TEAM_OF
from bazarblot.core.deal import Deal, DealResult, Phase
from bazarblot.core.dealing import deal_hands, deal_seed
from bazarblot.core.match import Match
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal

_RESAMPLE_STRIDE = 7_919  # a largish prime, keeps resampled trial seeds well-separated
_MAX_MATCH_DEALS = 100  # generous safety net — real matches to 301 finish in well under this


def play_deal_once(
    rules: RuleConfig,
    space: ActionSpace,
    seed: int,
    deal_number: int,
    dealer: int,
    agents: dict[int, Agent],
    match_score: tuple[int, int] = (0, 0),
) -> DealResult | None:
    """Play exactly one dealt hand at `(seed, deal_number)` — `redeal_attempt` is always 0, no
    retry — with `agents[seat]` making seat `seat`'s every decision. Returns `None` if the
    auction aborts (4 passes, no bid); see the module docstring for why the caller, not this
    function, decides what to do about that."""
    rng = random.Random(deal_seed(seed, deal_number, 0))
    hands = deal_hands(rng, rules)
    deal = Deal(rules, dealer=dealer, hands=hands, deal_id=deal_number)
    tracked = TrackedDeal(deal)
    while deal.phase in (Phase.AUCTION, Phase.PLAY):
        seat = deal.to_act
        info = info_set(tracked, seat, match_score=match_score, deal_number=deal_number)
        mask = legal_mask(info, space)
        action_idx = agents[seat].act(info, space, mask)
        tracked.step(space.decode(action_idx))
    if deal.phase == Phase.ABORTED:
        return None
    assert deal.result is not None
    return deal.result


@dataclass(frozen=True, slots=True)
class DuplicateResult:
    seed: int
    """The trial seed the pair actually completed on — may differ from the seed requested if
    earlier trials aborted and were discarded (see the module docstring)."""
    margin_x: float
    """Policy X's raw-score margin over Y, averaged across the two seat-swapped instances. Deal
    difficulty (an inherently high- or low-scoring shuffle) cancels almost entirely; what's left
    is mostly the skill difference between X and Y."""
    result_x_as_team0: DealResult
    result_x_as_team1: DealResult


def play_duplicate_pair(
    rules: RuleConfig,
    space: ActionSpace,
    seed: int,
    dealer: int,
    policy_x: Agent,
    policy_y: Agent,
    max_resample_attempts: int = 20,
) -> DuplicateResult | None:
    """Play the identical shuffle twice: X at seats {0, 2} (team 0) with Y at {1, 3} (team 1),
    then swapped. Returns `None` only if `max_resample_attempts` consecutive trial seeds all hit
    an abort on at least one side — astronomically unlikely with any reasonable agent."""
    agents_x_team0 = {0: policy_x, 2: policy_x, 1: policy_y, 3: policy_y}
    agents_x_team1 = {0: policy_y, 2: policy_y, 1: policy_x, 3: policy_x}

    for attempt in range(max_resample_attempts):
        trial_seed = seed + attempt * _RESAMPLE_STRIDE
        # Both runs are checked at every trial before deciding whether to resample — never
        # short-circuiting on the first one aborting — so that which policy happens to be
        # labeled "team0" doesn't change which trial_seed the pair ultimately settles on. Without
        # this, `play_duplicate_pair(seed, X, Y)` and `play_duplicate_pair(seed, Y, X)` could
        # silently land on different underlying deals whenever exactly one of the two seat
        # assignments aborts at a given trial.
        result_a = play_deal_once(rules, space, trial_seed, 0, dealer, agents_x_team0)
        result_b = play_deal_once(rules, space, trial_seed, 0, dealer, agents_x_team1)
        if result_a is None or result_b is None:
            continue

        team0_a, team1_a = result_a.team_scores
        margin_x_a = team0_a - team1_a  # X is team 0 in run A
        team0_b, team1_b = result_b.team_scores
        margin_x_b = team1_b - team0_b  # X is team 1 in run B

        return DuplicateResult(
            seed=trial_seed,
            margin_x=(margin_x_a + margin_x_b) / 2,
            result_x_as_team0=result_a,
            result_x_as_team1=result_b,
        )
    return None


def _play_match_sequence(
    rules: RuleConfig, space: ActionSpace, seed: int, agent_by_team: dict[int, Agent]
) -> Match | None:
    """One full match, deals numbered 0, 1, 2, ... in order, `redeal_attempt` always 0 — `None`
    the instant any deal aborts, rather than silently redealing it (see module docstring)."""
    match = Match(rules=rules)
    deal_number = 0
    while not match.finished:
        agents = {s: agent_by_team[TEAM_OF[s]] for s in range(4)}
        result = play_deal_once(
            rules,
            space,
            seed,
            deal_number,
            match.dealer,
            agents,
            match_score=(match.scores[0], match.scores[1]),
        )
        if result is None:
            return None
        match.apply_deal_result(result)
        deal_number += 1
        if deal_number > _MAX_MATCH_DEALS:
            raise RuntimeError(
                f"match seed={seed} did not terminate within {_MAX_MATCH_DEALS} deals"
            )
    return match


@dataclass(frozen=True, slots=True)
class DuplicateMatchResult:
    seed: int
    match_x_as_team0: Match
    match_x_as_team1: Match

    @property
    def x_win(self) -> float | None:
        """1.0 if X won both instances, 0.0 if X lost both, 0.5 if split, `None` if either
        instance's match ended undecided (the exact-tie-while-both-cross case `Match` itself
        documents as unresolved rather than guessed at)."""
        results = []
        if self.match_x_as_team0.winner is not None:
            results.append(1.0 if self.match_x_as_team0.winner == 0 else 0.0)
        if self.match_x_as_team1.winner is not None:
            results.append(1.0 if self.match_x_as_team1.winner == 1 else 0.0)
        if not results:
            return None
        return sum(results) / len(results)


def play_duplicate_match(
    rules: RuleConfig,
    space: ActionSpace,
    seed: int,
    policy_x: Agent,
    policy_y: Agent,
    max_resample_attempts: int = 50,
) -> DuplicateMatchResult | None:
    """Play a full match (to `rules.match.target`) twice with the identical sequence of dealt
    hands, X assigned team 0 in one run and team 1 in the other — the match-level analogue of
    `play_duplicate_pair`. Resamples the WHOLE match's seed (not per-deal) on any abort, so
    pairing stays exact across the entire match rather than only approximately holding. `None` if
    no clean seed was found within `max_resample_attempts` (matches are many deals long, so this
    needs more attempts than the single-deal case to be comparably unlikely to ever trigger).

    Unlike `play_duplicate_pair`, this DOES short-circuit on `match_a` aborting without also
    simulating `match_b` at that trial — deliberately, since each `_play_match_sequence` call is
    many deals deep and doubling that cost on every trial (most of which succeed) isn't worth it
    just to make `play_duplicate_match(seed, X, Y)` and `play_duplicate_match(seed, Y, X)`
    resample in lockstep, a property nothing here currently depends on or tests."""
    for attempt in range(max_resample_attempts):
        trial_seed = seed + attempt * _RESAMPLE_STRIDE
        match_a = _play_match_sequence(rules, space, trial_seed, {0: policy_x, 1: policy_y})
        if match_a is None:
            continue
        match_b = _play_match_sequence(rules, space, trial_seed, {0: policy_y, 1: policy_x})
        if match_b is None:
            continue
        return DuplicateMatchResult(
            seed=trial_seed, match_x_as_team0=match_a, match_x_as_team1=match_b
        )
    return None
