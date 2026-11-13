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
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from bazarblot.agents.base import Agent
from bazarblot.agents.nn.network import (
    OUTCOME_ABORT,
    OUTCOME_THEY_FAIL,
    OUTCOME_THEY_MAKE,
    OUTCOME_WE_FAIL,
    OUTCOME_WE_MAKE,
    CentralCritic,
    PolicyValueNet,
)
from bazarblot.agents.nn.ppo_agent import Decision, decide
from bazarblot.core.cards import TEAM_OF
from bazarblot.core.deal import Deal, DealResult, Phase
from bazarblot.core.dealing import deal_hands, deal_seed
from bazarblot.core.rules import RuleConfig
from bazarblot.env.actions import ActionSpace, legal_mask
from bazarblot.env.canonical import canonical_belief_target
from bazarblot.env.infoset import info_set
from bazarblot.env.tracked_deal import TrackedDeal
from bazarblot.learn.belief import belief_target, privileged_features
from bazarblot.learn.joint_search import JointSearch, JointSearchConfig
from bazarblot.learn.reward import SQUASH_SCALE, deal_margin_reward

Opponent = PolicyValueNet | Agent


@dataclass(slots=True)
class Transition:
    decision: Decision
    belief_target: npt.NDArray[np.int8] | None = None
    """Ground-truth holder of every unseen card at this decision (`learn/belief.py`), computed
    from the deal in rollout code -- training-only, never part of the policy's input."""
    outcome_target: int = -1
    """How the deal ended from this seat's view (`OUTCOME_*` in `agents/nn/network.py`), filled in
    once the deal is over; -1 = no label. Training-only, like `belief_target`."""
    value: float = 0.0
    """Baseline used for the advantage: the actor's own value head by default, or the centralized
    critic's estimate when one is supplied to `collect_rollout`."""
    reward: float = 0.0
    trick_points: float = 0.0
    """Card points of the trick this card was played to, + if this seat's team won it, - if the
    opponents did (last-trick bonus included). Used only with `trick_credit` (see
    `play_rollout_deals`)."""
    advantage: float = 0.0
    ret: float = 0.0


def _gae(
    transitions: list[Transition], gamma: float, lam: float, lam_auction: float | None = None
) -> None:
    """In-place GAE over one seat's own sub-trajectory within a single deal -- the trajectory
    truly ends when the deal does, so there is no bootstrap past the last transition
    (`next_value = 0` at the start of the backward pass)."""
    next_value = 0.0
    gae = 0.0
    for t in reversed(transitions):
        delta = t.reward + gamma * next_value - t.value
        # `lam_auction` (e.g. 0) cuts the auction decisions' credit off at the next decision:
        # a bid is judged by the critic's value of the position it leads to -- ultimately the
        # contract reached, at the seat's first card -- not by how the cards were then played.
        step_lam = lam if lam_auction is None or t.decision.phase == Phase.PLAY else lam_auction
        gae = delta + gamma * step_lam * gae
        t.advantage = gae
        t.ret = gae + t.value
        next_value = t.value


def _assign_critic_values(critic: CentralCritic, transitions: list[Transition]) -> None:
    """Overwrite each transition's baseline `value` with the centralized critic's estimate, in
    batches of a few thousand under `no_grad`."""
    was_training = critic.training
    critic.eval()
    with torch.no_grad():
        for start in range(0, len(transitions), 4096):
            chunk = transitions[start : start + 4096]
            obs = torch.from_numpy(np.stack([t.decision.obs for t in chunk]))
            targets = np.stack([_require_belief_target(t) for t in chunk])
            priv = torch.from_numpy(privileged_features(targets))
            values = critic(obs, priv)
            for t, v in zip(chunk, values.tolist(), strict=True):
                t.value = float(v)
    critic.train(was_training)


def _require_belief_target(t: Transition) -> npt.NDArray[np.int8]:
    if t.belief_target is None:
        raise ValueError("a centralized critic needs ground-truth hands on every transition")
    return t.belief_target


@dataclass(slots=True)
class RolloutStats:
    n_deals: int = 0
    n_aborted: int = 0
    n_transitions: int = 0
    n_opponent_deals: int = 0
    n_contested_completed: int = 0
    n_learner_declared_contested: int = 0
    n_capped: int = 0
    """Auctions stopped by the engine's `max_auction_steps` safety cap (scored as aborts)."""
    raw_margins: list[float] = field(default_factory=list)
    selfplay_levels: list[int] = field(default_factory=list)
    """Contract level of every completed pure-self-play deal (all four seats the learner)."""
    n_selfplay_made: int = 0
    n_selfplay_doubled: int = 0
    n_searches: int = 0
    """Joint bid searches run (`learn/joint_search.py`), and how many changed the bid / the
    partner's reply."""
    n_search_deviations: int = 0
    n_search_replies: int = 0
    n_search_reply_changes: int = 0

    @property
    def learner_declare_rate(self) -> float | None:
        """Fraction of COMPLETED opponent-mix deals where the learner's own team won the
        auction, i.e. actually chose to bid rather than concede it. `None` when no opponent-mix
        deal completed this rollout to measure it from.

        Deliberately excludes pure self-play deals: there, "who declared" is uninformative (both
        teams are the identical network, so it says nothing about skill), and PURE self-play's
        own abort rate can be near 100% even for a healthy policy playing a FIXED opponent well
        (see `docs/03-implementation-roadmap.md` M7: the checkpoint that beat `heuristic`/`random`
        turned out to never once declare against them -- a pure pass-and-defend strategy -- and
        this is the stat that would have caught it during training instead of only after)."""
        if self.n_contested_completed == 0:
            return None
        return self.n_learner_declared_contested / self.n_contested_completed


def _play_one(
    rules: RuleConfig,
    space: ActionSpace,
    seed: int,
    dealer: int,
    learner: PolicyValueNet,
    seat_actors: dict[int, Opponent],
    forced_bid_prob: float = 0.0,
    bid_agent: Agent | None = None,
    forced_contra_prob: float = 0.0,
    joint_search: JointSearch | None = None,
    search_rng: random.Random | None = None,
) -> tuple[dict[int, list[Transition]], float, bool, DealResult | None, bool]:
    """Play one deal; `seat_actors[seat]` decides seat `seat`'s actions -- either `learner`
    itself, a frozen `PolicyValueNet` snapshot from the opponent pool, or an arbitrary `Agent`
    (a fixed baseline like `HeuristicAgent`, see `collect_rollout`'s docstring on why the pool
    can hold both). Returns `(transitions by learner seat, unsquashed margin for team 0,
    aborted, result, capped)` -- `result` is `None` when the deal aborted (no contract at all).
    `capped` is True when the auction hit the engine's `max_auction_steps` safety cap: an
    untrained network that bids about half the time can keep an auction circling, and that is
    scored as an abort (margin 0) rather than crashing the run; `RolloutStats.n_capped` makes it
    visible. `forced_bid_prob` is applied only to the LEARNER's own auction decisions (see
    `agents/nn/ppo_agent.py::_auction_decision`) -- opponents play unperturbed, exactly as
    they would for real.

    `bid_agent`, when given, makes EVERY auction decision for EVERY seat (learner and opponents
    alike) and no auction transition is recorded: the learner is trained on card play only, under
    auctions it cannot influence. This is the first stage of the plan to fix the agent never
    bidding (`docs/03-implementation-roadmap.md` M7): with bidding held fixed there is no
    pass-everything shortcut to fall into, and play skill can be measured on its own.

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
    # Joint bid search runs only in pure self-play deals (all four seats the learner).
    searching = joint_search is not None and len(transitions) == 4 and search_rng is not None
    reply_due: set[int] = set()  # partners owed a best reply to a searched deviation

    while deal.phase in (Phase.AUCTION, Phase.PLAY):
        if (
            deal.phase == Phase.AUCTION
            and len(tracked.auction_log) >= rules.auction.max_auction_steps
        ):
            return transitions, 0.0, True, None, True
        seat = deal.to_act
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        mask = legal_mask(info, space)
        actor = seat_actors[seat]
        if bid_agent is not None and info.phase == Phase.AUCTION:
            action_idx = bid_agent.act(info, space, mask)
        elif actor is learner:
            force: int | None = None
            if searching and info.phase == Phase.AUCTION:
                assert joint_search is not None and search_rng is not None
                if seat in reply_due:
                    reply_due.discard(seat)
                    force = joint_search.best_reply(tracked)
                elif search_rng.random() < joint_search.config.prob:
                    force = joint_search.search(tracked)
                    if force is not None:
                        reply_due.add((seat + 2) % 4)
            decision = decide(
                learner,
                info,
                space,
                mask,
                sample=True,
                forced_bid_prob=forced_bid_prob,
                forced_contra_prob=forced_contra_prob,
                force_env_action=force,
            )
            transitions[seat].append(
                Transition(
                    decision=decision,
                    belief_target=canonical_belief_target(
                        belief_target(deal.hands, seat), decision.perm
                    ),
                    value=decision.value,
                )
            )
            action_idx = decision.env_action_idx
        elif isinstance(actor, PolicyValueNet):
            action_idx = decide(actor, info, space, mask, sample=True).env_action_idx
        else:
            action_idx = actor.act(info, space, mask)
        n_tricks = len(deal.tricks)
        tracked.step(space.decode(action_idx))
        if len(deal.tricks) > n_tricks:  # a trick just completed: note it on each learner's card
            trick = deal.tricks[-1]
            points = trick.points
            if len(deal.tricks) == rules.deal.cards_per_player:
                points += rules.contracts.last_hand_bonus
            for s, seat_transitions in transitions.items():
                if seat_transitions and seat_transitions[-1].decision.phase == Phase.PLAY:
                    same_team = TEAM_OF[s] == TEAM_OF[trick.winner]
                    seat_transitions[-1].trick_points += points if same_team else -points

    if deal.phase == Phase.ABORTED:
        return transitions, 0.0, True, None, False
    assert deal.result is not None
    team0, team1 = deal.result.team_scores
    return transitions, float(team0 - team1), False, deal.result, False


def outcome_label(seat: int, result: DealResult | None) -> int:
    """The contract-outcome class of a finished deal from `seat`'s point of view."""
    if result is None:
        return OUTCOME_ABORT
    if TEAM_OF[seat] == result.attackers_team:
        return OUTCOME_WE_MAKE if result.made else OUTCOME_WE_FAIL
    return OUTCOME_THEY_MAKE if result.made else OUTCOME_THEY_FAIL


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
    forced_bid_prob: float = 0.0,
    bid_agent: Agent | None = None,
    critic: CentralCritic | None = None,
    forced_contra_prob: float = 0.0,
    reward_scale: float = SQUASH_SCALE,
    trick_credit: bool = False,
    lam_auction: float | None = None,
    search_config: JointSearchConfig | None = None,
) -> tuple[list[Transition], RolloutStats]:
    """Collect `n_deals` shuffles' worth of self-play data for `learner`.

    `critic` (a training-only centralized value function that sees the hidden hands, spec §7.4)
    replaces the actor's own value head as the baseline for the advantages: after the whole batch
    of deals is collected, every transition's value is re-estimated by the critic in a few batched
    calls and only then are the advantages computed. The actor never sees what the critic sees.

    `forced_bid_prob` (see `agents/nn/ppo_agent.py::_auction_decision`) biases the LEARNER's own
    auction sampling toward `BID` during collection, without touching what `legal_mask` itself
    says is legal and without corrupting the recorded log-probabilities PPO's update depends on.
    Zero by default (no effect); `learn/train.py` anneals it down over a training run.

    `forced_contra_prob` does the same for doubling (contra/recontra) when it is legal.

    `reward_scale` is the `tanh(margin / scale)` squash constant (`learn/reward.py`); larger values
    saturate later, so large-stakes outcomes (a doubled failure) stay distinguishable.

    `bid_agent` (see `_play_one`) takes over all auction decisions so the learner trains on card
    play alone; the forced-exploration probabilities are then moot.

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
    trajectories, stats = play_rollout_deals(
        rules,
        space,
        learner,
        range(n_deals),
        seed,
        rng,
        opponent_pool=opponent_pool,
        opponent_prob=opponent_prob,
        forced_bid_prob=forced_bid_prob,
        bid_agent=bid_agent,
        forced_contra_prob=forced_contra_prob,
        reward_scale=reward_scale,
        trick_credit=trick_credit,
        search_config=search_config,
        critic=critic,
    )
    return finalize_rollout(trajectories, critic, gamma, lam, lam_auction), stats


def play_rollout_deals(
    rules: RuleConfig,
    space: ActionSpace,
    learner: PolicyValueNet,
    deal_indices: Sequence[int],
    seed: int,
    rng: random.Random,
    opponent_pool: list[Opponent] | None = None,
    opponent_prob: float = 0.0,
    forced_bid_prob: float = 0.0,
    bid_agent: Agent | None = None,
    forced_contra_prob: float = 0.0,
    reward_scale: float = SQUASH_SCALE,
    trick_credit: bool = False,
    search_config: JointSearchConfig | None = None,
    critic: CentralCritic | None = None,
) -> tuple[list[list[Transition]], RolloutStats]:
    """The deal-playing half of `collect_rollout`: plays shuffle `i` (deal seed `seed + i`,
    dealer `i % 4`) for every `i` in `deal_indices` and returns each learner seat's trajectory
    with terminal rewards and outcome labels filled in, but no critic values or advantages yet
    (`finalize_rollout`). Split out so `learn/parallel.py` can run it in worker processes on
    disjoint index ranges; `rng` decides opponent sampling."""
    trajectories: list[list[Transition]] = []
    stats = RolloutStats()
    joint_search = (
        JointSearch(rules, space, learner, critic, search_config)
        if search_config is not None and search_config.prob > 0
        else None
    )
    search_rng = random.Random(rng.random())

    for i in deal_indices:
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
            transitions, margin_team0, aborted, result, capped = _play_one(
                rules,
                space,
                deal_seed_value,
                dealer,
                learner,
                seat_actors,
                forced_bid_prob,
                bid_agent,
                forced_contra_prob,
                joint_search,
                search_rng,
            )
            stats.n_deals += 1
            stats.n_capped += int(capped)
            if result is not None and not use_opponent:
                stats.selfplay_levels.append(result.contract.level)
                stats.n_selfplay_made += int(result.made)
                stats.n_selfplay_doubled += int(result.contract.doubling != "none")
            if aborted:
                stats.n_aborted += 1
            elif use_opponent:
                assert result is not None
                stats.n_contested_completed += 1
                learner_team = 0 if seat_actors[0] is learner else 1
                if result.attackers_team == learner_team:
                    stats.n_learner_declared_contested += 1
            stats.raw_margins.append(margin_team0)
            for seat, seat_transitions in transitions.items():
                if not seat_transitions:
                    continue
                margin = margin_team0 if TEAM_OF[seat] == 0 else -margin_team0
                final = deal_margin_reward(margin, reward_scale)
                if trick_credit:
                    # Each card is credited with its trick's points as they happen, in the
                    # reward's units (10 card points = 1 score point, divided by the squash scale,
                    # the slope of tanh at 0); the last reward is reduced by the same total, so the
                    # seat's return is unchanged and so is the best policy.
                    k = 1.0 / (10.0 * reward_scale)
                    for t in seat_transitions:
                        t.reward = k * t.trick_points
                    seat_transitions[-1].reward += final - sum(
                        k * t.trick_points for t in seat_transitions
                    )
                else:
                    seat_transitions[-1].reward = final
                label = outcome_label(seat, result)
                for t in seat_transitions:
                    t.outcome_target = label
                trajectories.append(seat_transitions)
                stats.n_transitions += len(seat_transitions)

    if joint_search is not None:
        stats.n_searches = joint_search.stats.searches
        stats.n_search_deviations = joint_search.stats.deviations
        stats.n_search_replies = joint_search.stats.replies
        stats.n_search_reply_changes = joint_search.stats.reply_changes
    return trajectories, stats


def finalize_rollout(
    trajectories: list[list[Transition]],
    critic: CentralCritic | None,
    gamma: float,
    lam: float,
    lam_auction: float | None = None,
) -> list[Transition]:
    """Baselines (the centralized critic's, if given) and per-trajectory GAE; returns the flat
    transition list in trajectory order."""
    all_transitions = [t for traj in trajectories for t in traj]
    if critic is not None and all_transitions:
        _assign_critic_values(critic, all_transitions)
    for seat_transitions in trajectories:
        _gae(seat_transitions, gamma, lam, lam_auction)
    return all_transitions


def merge_stats(parts: Sequence[RolloutStats]) -> RolloutStats:
    """Combine the stats of several `play_rollout_deals` calls (e.g. one per worker)."""
    out = RolloutStats()
    for p in parts:
        out.n_deals += p.n_deals
        out.n_aborted += p.n_aborted
        out.n_transitions += p.n_transitions
        out.n_opponent_deals += p.n_opponent_deals
        out.n_contested_completed += p.n_contested_completed
        out.n_learner_declared_contested += p.n_learner_declared_contested
        out.n_capped += p.n_capped
        out.raw_margins.extend(p.raw_margins)
        out.selfplay_levels.extend(p.selfplay_levels)
        out.n_selfplay_made += p.n_selfplay_made
        out.n_selfplay_doubled += p.n_selfplay_doubled
        out.n_searches += p.n_searches
        out.n_search_deviations += p.n_search_deviations
        out.n_search_replies += p.n_search_replies
        out.n_search_reply_changes += p.n_search_reply_changes
    return out
