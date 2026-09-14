"""Self-play rollout collection for PPO (M7): plays deals with the current network occupying
either all four seats (pure self-play) or two seats against a frozen opponent-pool snapshot,
records every decision the CURRENT (learner) network made as a `Transition`, and computes GAE
advantages/returns per seat's own sub-trajectory once the deal's terminal `deal_margin` reward is
known.

**Why per-seat, not per-deal, trajectories.** A deal has up to 32 plies split across four seats;
from any one seat's point of view, only ITS OWN decisions matter as an MDP -- the other three
seats' moves are part of the environment's transition dynamics, exactly the simplification every
self-play PPO/DMC card-game agent makes (spec §7.1 family 1; the alternative, a fully centralized
multi-agent formulation, is family 3/4's territory, not the baseline). Each seat's own decisions,
in order, are treated as one short single-agent trajectory that ends -- no bootstrap past it --
when the deal does, with reward 0 at every step except the last, which carries that seat's team's
`deal_margin`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from bazarblot.agents.base import Agent
from bazarblot.agents.nn.network import PolicyValueNet
from bazarblot.agents.nn.ppo_agent import Decision, decide
from bazarblot.core.cards import TEAM_OF
from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands, deal_seed
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal
from bazarblot.learn.reward import deal_margin_reward

Opponent = PolicyValueNet | Agent


@dataclass(slots=True)
class Transition:
    decision: Decision
    reward: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


def _gae(transitions: list[Transition], gamma: float, lam: float) -> None:
    """In-place GAE over one seat's own sub-trajectory within a single deal -- the trajectory
    truly ends when the deal does, so there is no bootstrap past the last transition
    (`next_value = 0` at the start of the backward pass)."""
    next_value = 0.0
    gae = 0.0
    for t in reversed(transitions):
        delta = t.reward + gamma * next_value - t.decision.value
        gae = delta + gamma * lam * gae
        t.advantage = gae
        t.ret = gae + t.decision.value
        next_value = t.decision.value


@dataclass(slots=True)
class RolloutStats:
    n_deals: int = 0
    n_aborted: int = 0
    n_transitions: int = 0
    n_opponent_deals: int = 0
    raw_margins: list[float] = field(default_factory=list)


def _play_one(
    rules: RuleConfig,
    space: ActionSpace,
    seed: int,
    dealer: int,
    learner: PolicyValueNet,
    seat_actors: dict[int, Opponent],
) -> tuple[dict[int, list[Transition]], float, bool]:
    """Play one deal; `seat_actors[seat]` decides seat `seat`'s actions -- either `learner`
    itself, a frozen `PolicyValueNet` snapshot from the opponent pool, or an arbitrary `Agent`
    (a fixed baseline like `HeuristicAgent`, see `collect_rollout`'s docstring on why the pool
    can hold both). Returns `(transitions by learner seat, unsquashed margin for team 0,
    aborted)`.

    **A 4-pass abort is scored as a real, zero-margin outcome -- it is NOT discarded**, unlike
    `eval/duplicate.py`, which discards and resamples specifically to preserve its "identical
    shuffle, seats swapped" pairing guarantee (a property training data doesn't need). An earlier
    version of this function discarded aborts here too, by analogy -- and a real training run hit
    a self-play collapse because of exactly that choice (see `docs/03-implementation-roadmap.md`
    M7): discarding an aborted deal makes it invisible to the gradient either way, so once per-seat
    pass probability drifts up even slightly (ordinary policy-gradient noise, amplified because
    abort probability is roughly `P(pass)^4` across four IDENTICAL seats in self-play), nothing
    pulls it back down -- the only deals still producing a training signal are the shrinking
    minority that don't abort. Scoring an abort as `margin=0` isn't reward shaping: under the
    rules an abort truly is zero-sum-zero (nobody scores, the hand is simply redealt), so this is
    the TRUE terminal value of that outcome, not an invented one, and it restores a real
    corrective gradient (passing on a hand the value function has learned is worth opening now
    gets a negative advantage, rather than no signal at all)."""
    rng = random.Random(deal_seed(seed, 0, 0))
    hands = deal_hands(rng, rules)
    deal = Deal(rules, dealer=dealer, hands=hands, deal_id=0)
    tracked = TrackedDeal(deal)
    transitions: dict[int, list[Transition]] = {
        s: [] for s in seat_actors if seat_actors[s] is learner
    }

    while deal.phase in (Phase.AUCTION, Phase.PLAY):
        seat = deal.to_act
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        mask = legal_mask(info, space)
        actor = seat_actors[seat]
        if actor is learner:
            decision = decide(learner, info, space, mask, sample=True)
            transitions[seat].append(Transition(decision=decision))
            action_idx = decision.action_idx
        elif isinstance(actor, PolicyValueNet):
            action_idx = decide(actor, info, space, mask, sample=True).action_idx
        else:
            action_idx = actor.act(info, space, mask)
        tracked.step(space.decode(action_idx))

    if deal.phase == Phase.ABORTED:
        return transitions, 0.0, True
    assert deal.result is not None
    team0, team1 = deal.result.team_scores
    return transitions, float(team0 - team1), False


def collect_rollout(
    rules: RuleConfig,
    space: ActionSpace,
    learner: PolicyValueNet,
    n_deals: int,
    seed: int,
    opponent_pool: list[Opponent] | None = None,
    opponent_prob: float = 0.0,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[list[Transition], RolloutStats]:
    """Collect `n_deals` shuffles' worth of self-play data for `learner`.

    Each shuffle is either pure self-play (all four seats `learner`) or, with probability
    `opponent_prob` when `opponent_pool` is non-empty, `learner` at two seats against one
    randomly-sampled pool member at the other two -- played TWICE, once with `learner` as team 0
    and once (same shuffle, via `deal_seed`'s determinism) as team 1, so deal luck cancels within
    this batch rather than only across many batches (spec §6, "paired sampling within the
    batch"). Pure self-play shuffles need no such pairing -- both teams are the identical
    network, so they are already symmetric.

    **`opponent_pool` deliberately accepts both frozen `PolicyValueNet` snapshots (the "league"
    spec §7.1 asks for) and arbitrary fixed `Agent`s like `HeuristicAgent`/`RandomAgent`.** A
    pool seeded with a fixed, always-reasonable bidder from iteration 0 -- not just historical
    self-play snapshots that only start existing after the first `snapshot_every` iterations --
    matters a lot in practice: pure self-play among four IDENTICAL, still-randomly-initialized
    seats can collapse to a degenerate "everyone always passes" equilibrium within a handful of
    updates (every auction aborts, margin stuck at exactly 0, entropy decaying toward 0), because
    that outcome is a genuine, self-consistent zero-reward fixed point of the symmetric game and
    nothing forces the policy to explore away from it once found (see `docs/03-...` M7 for the
    real run that hit this). Mixing in a fixed baseline that reliably DOES open contested
    auctions breaks the symmetry and keeps real bid/play outcomes flowing into training data
    throughout, independent of what the learner's own policy has drifted toward."""
    rng = random.Random(seed)
    all_transitions: list[Transition] = []
    stats = RolloutStats()

    for i in range(n_deals):
        dealer = i % 4
        deal_seed_value = seed + i
        use_opponent = bool(opponent_pool) and rng.random() < opponent_prob

        seat_assignments: tuple[dict[int, Opponent], ...]
        if use_opponent:
            assert opponent_pool is not None
            opponent = opponent_pool[rng.randrange(len(opponent_pool))]
            seat_assignments = (
                {0: learner, 2: learner, 1: opponent, 3: opponent},
                {0: opponent, 2: opponent, 1: learner, 3: learner},
            )
            stats.n_opponent_deals += 1
        else:
            seat_assignments = ({0: learner, 1: learner, 2: learner, 3: learner},)

        for seat_actors in seat_assignments:
            transitions, margin_team0, aborted = _play_one(
                rules, space, deal_seed_value, dealer, learner, seat_actors
            )
            stats.n_deals += 1
            if aborted:
                stats.n_aborted += 1
            stats.raw_margins.append(margin_team0)
            for seat, seat_transitions in transitions.items():
                if not seat_transitions:
                    continue
                margin = margin_team0 if TEAM_OF[seat] == 0 else -margin_team0
                seat_transitions[-1].reward = deal_margin_reward(margin)
                _gae(seat_transitions, gamma, lam)
                all_transitions.extend(seat_transitions)
                stats.n_transitions += len(seat_transitions)

    return all_transitions, stats
