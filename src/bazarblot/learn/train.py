"""The M7 self-play PPO training loop: collect rollout -> PPO update -> periodically snapshot into
the opponent pool -> periodically evaluate against `RandomAgent`/`HeuristicAgent` via `eval/`'s
paired harness (the same one M5 built and every other milestone's numbers already trust) ->
checkpoint. `Done when` (spec/roadmap M7): the learned agent beats `heuristic` with a paired CI
excluding 0 -- `TrainConfig.eval_every` iterations, this loop checks exactly that and logs it, so
"is M7 done yet" is a number in the log, not a guess.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from bazarblot.agents.base import Agent
from bazarblot.agents.composed import ComposedAgent
from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.network import CentralCritic, apply_action_prior, build_network
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.agents.pass_bidder import PassBidder
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.deal import Phase
from bazarblot.core.rules import RuleConfig, load_default
from bazarblot.env.actions import ActionSpace, build_action_space
from bazarblot.eval.behavior import evaluate_behavior
from bazarblot.eval.metrics import evaluate_pairs
from bazarblot.learn.checkpoint import critic_from_payload, load_checkpoint, save_checkpoint
from bazarblot.learn.opponent_pool import OpponentPool
from bazarblot.learn.ppo import PPOConfig, ppo_update
from bazarblot.learn.rollout import Opponent, collect_rollout
from bazarblot.learn.signaling import measure_signaling


@dataclass(frozen=True, slots=True)
class TrainConfig:
    n_iterations: int = 1000
    n_deals_per_iter: int = 256
    seed: int = 0
    hidden: tuple[int, ...] = (256, 256)
    gamma: float = 1.0
    lam: float = 0.95
    reward_scale: float = 24.0
    """Constant in the `tanh(margin / reward_scale)` reward squash (spec §6). 24 is the spec's
    recommendation; a larger value saturates later. Changing it rescales the value function, so
    runs with different values are not comparable on loss curves, only on behaviour."""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.02
    forced_bid_prob_start: float = 0.4
    forced_bid_prob_end: float = 0.05
    forced_bid_anneal_frac: float = 1.0
    forced_contra_prob_start: float = 0.0
    forced_contra_prob_end: float = 0.0
    """Exploration of doubling (contra/recontra) during rollouts, linearly annealed over the run:
    each time doubling is legal for a learner seat, take it with this probability. A policy cloned
    from a teacher that never doubles assigns it near-zero probability and PPO cannot learn its
    value from actions it never takes (`docs/03-implementation-roadmap.md` M7)."""
    bid_agent: str | None = None
    """`"heuristic"` trains card play only, with `HeuristicAgent` making every auction decision for
    every seat (roadmap M7 stage 1); `None` is the original joint bid-and-play training. Evaluation
    then pits `heuristic bidding + learned play` against `heuristic bidding + baseline play`, so a
    margin reflects play skill alone and cannot come from conceding auctions."""
    raise_prior_decay: float = 0.0
    capot_prior_bias: float = 0.0
    double_prior_bias: float = 0.0
    """Starting biases of the auction heads for a network trained from scratch
    (`agents/nn/network.py::apply_action_prior`): a prior on the shape of the action space (small
    raises, rare capot, rare doubling), not on which hands to bid. Ignored with
    `init_checkpoint`. All 0 reproduces the uniform start."""
    init_checkpoint: Path | None = None
    """Start from this checkpoint (e.g. the behavior-cloned player from `cli/imitate.py`) instead
    of a randomly initialised network."""
    value_warmup_iters: int = 0
    """For this many initial iterations, fit ONLY the value head (policy, entropy and belief
    terms off, optimizer over `value_head` parameters alone) on rollouts from the loaded policy.
    A cloned policy comes with an untrained critic; letting policy-gradient steps run against a
    garbage baseline would wreck the very policy the clone provides, so the critic is brought up
    to speed first with the policy frozen."""
    central_critic: bool = False
    """Use a centralized critic that sees the hidden hands (training only) as the advantage
    baseline instead of the actor's own value head (spec §7.4). The actor is unchanged and never
    sees what the critic sees."""
    critic_hidden: tuple[int, ...] = (256, 256)
    baseline_opponents: tuple[str, ...] = ("heuristic", "random")
    """Fixed opponents always in the training pool. Any of: `heuristic`; `heuristic@K` (the same
    heuristic with every bid target shifted by K levels, K<0 timid, K>0 aggressive); `random`;
    `passer` (always-pass bidder with heuristic card play); `clone` (a frozen copy of
    `init_checkpoint`, i.e. the network training started from)."""
    heldout_offsets: tuple[int, ...] = ()
    """If non-empty, every evaluation also plays the agent against heuristics at these bid offsets
    and logs the profile. Keep them DISJOINT from the offsets used in `baseline_opponents`, or the
    profile measures training opponents instead of generalisation."""
    heldout_pairs: int = 100
    signaling_deals: int = 200
    """Greedy self-play deals for the signaling probe at every evaluation
    (`learn/signaling.py`); 0 disables it. Skipped with `bid_agent`, whose auctions are not the
    learner's."""
    opponent_prob: float = 0.3
    pool_max_size: int = 8
    snapshot_every: int = 20
    eval_every: int = 20
    eval_n_pairs: int = 150
    checkpoint_dir: Path = Path("checkpoints/m7")
    log_dir: Path = Path("runs/m7")


def _entropy_coef_at(iteration: int, config: TrainConfig) -> float:
    """Linear anneal from `entropy_coef_start` down to `entropy_coef_end` over the run.

    The first M7 training run used a small FIXED `entropy_coef=0.01` and, despite two separate
    fixes for the pass-collapse pathology (see `learn/rollout.py`'s docstring and
    `docs/03-implementation-roadmap.md` M7), still drifted to a pure pass-and-defend policy that
    never once declared against `heuristic`/`random` in evaluation -- entropy fell from ~1.0 to
    ~0.2-0.3 within the first ~100 iterations and never recovered. A higher, ANNEALED entropy
    bonus keeps meaningful exploration pressure on the auction's meta-head for longer, rather than
    letting it collapse this early; annealing down (rather than holding the higher value for the
    whole run) still lets the policy sharpen into a decisive one by the end, which a constant high
    bonus would fight against indefinitely."""
    if config.n_iterations <= 1:
        return config.entropy_coef_start
    frac = iteration / (config.n_iterations - 1)
    return config.entropy_coef_start + frac * (config.entropy_coef_end - config.entropy_coef_start)


def _forced_bid_prob_at(iteration: int, config: TrainConfig) -> float:
    """Linear anneal from `forced_bid_prob_start` down to `forced_bid_prob_end` over the first
    `forced_bid_anneal_frac` of the run, then hold at `forced_bid_prob_end` for the rest.

    Added after entropy annealing ALONE (`_entropy_coef_at`) failed to prevent the pass-collapse
    even with a much higher, more slowly-decaying bonus (0.05 -> 0.02 over 800 iterations):
    `learner_declare_rate` (`learn/rollout.py::RolloutStats`) still fell to ~0 within the first
    ~10-15 iterations and stayed there, unchanged from the earlier, un-annealed failure.

    **`forced_bid_prob_end` defaults to a small nonzero floor (0.05), not 0.** A first version of
    this schedule annealed all the way to 0 by 60% of the run -- and a diagnostic run showed
    `learner_declare_rate` tracking `forced_bid_prob` almost exactly (both ~0.4 early, both ~0 by
    the time forcing hit 0), then staying at ~0 for the remainder even though `value_loss` and
    `entropy` kept moving throughout, i.e. the policy kept learning something, just never that
    bidding pays. That is consistent with a real, not merely exploration-starved, dynamic:
    declaring only pays off with competent card play, early-training play skill is still poor
    (learned largely from defending, which needs no bids at all), so the value function correctly
    learns from the forced attempts that bidding is a losing proposition FOR THE CURRENT POLICY --
    and once forcing stops, there is no more fresh bid data to let it re-evaluate that conclusion
    as play skill matures later. A permanent small floor keeps real bid outcomes flowing into
    training data for the whole run, so the value function can keep reassessing profitability
    instead of freezing its verdict early; it does not force the FINAL policy to bid (evaluation
    and `decide(..., sample=False)` are greedy and ignore `forced_bid_prob` entirely -- see
    `agents/nn/ppo_agent.py::decide`), it only keeps exploration alive during training."""
    if config.forced_bid_anneal_frac <= 0 or config.n_iterations <= 1:
        return config.forced_bid_prob_end
    anneal_iters = max(1, int(config.n_iterations * config.forced_bid_anneal_frac))
    if iteration >= anneal_iters:
        return config.forced_bid_prob_end
    frac = iteration / anneal_iters
    return config.forced_bid_prob_start + frac * (
        config.forced_bid_prob_end - config.forced_bid_prob_start
    )


def _forced_contra_prob_at(iteration: int, config: TrainConfig) -> float:
    """Linear anneal from `forced_contra_prob_start` to `forced_contra_prob_end` over the run."""
    if config.n_iterations <= 1:
        return config.forced_contra_prob_start
    frac = iteration / (config.n_iterations - 1)
    return config.forced_contra_prob_start + frac * (
        config.forced_contra_prob_end - config.forced_contra_prob_start
    )


def _log_eval(writer: SummaryWriter, tag: str, report: object, iteration: int) -> None:
    from bazarblot.eval.metrics import DuplicateEvalReport

    assert isinstance(report, DuplicateEvalReport)
    writer.add_scalar(f"{tag}/margin_mean", report.margin.mean, iteration)
    writer.add_scalar(f"{tag}/margin_ci_lo", report.margin.lo, iteration)
    writer.add_scalar(f"{tag}/margin_ci_hi", report.margin.hi, iteration)
    beats = report.margin.lo > 0.0
    writer.add_scalar(f"{tag}/beats_with_ci_excluding_zero", float(beats), iteration)
    print(
        f"  [eval:{tag}] margin={report.margin.mean:.2f} "
        f"CI=({report.margin.lo:.2f}, {report.margin.hi:.2f}) "
        f"beats_baseline={beats} n_pairs={report.n_pairs}"
    )


def _build_opponent(
    name: str,
    rules: RuleConfig,
    config: TrainConfig,
    heuristic_agent: Agent,
    random_agent: Agent,
    passer_agent: Agent,
) -> Opponent:
    """Resolve one `TrainConfig.baseline_opponents` entry to an opponent."""
    if name == "heuristic":
        return heuristic_agent
    if name == "random":
        return random_agent
    if name == "passer":
        return passer_agent
    if name.startswith("heuristic@"):
        try:
            offset = int(name.split("@", 1)[1])
        except ValueError:
            raise ValueError(f"bad heuristic offset in opponent {name!r}") from None
        return HeuristicAgent(rules, bid_offset=offset)
    if name == "clone":
        if config.init_checkpoint is None:
            raise ValueError("opponent 'clone' needs init_checkpoint (it is a frozen copy of it)")
        frozen, _ = load_checkpoint(config.init_checkpoint, rules)
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        return frozen
    raise ValueError(
        f"unknown baseline opponents [{name!r}]; expected heuristic, heuristic@K, random, "
        "passer or clone"
    )


def _heldout_profile(
    rules: RuleConfig,
    space: ActionSpace,
    agent: Agent,
    offsets: tuple[int, ...],
    seeds: list[int],
    dealers: list[int],
) -> dict[int, float]:
    """Paired margin of `agent` over `HeuristicAgent(bid_offset=k)` for each held-out offset."""
    profile: dict[int, float] = {}
    for k in offsets:
        opponent = HeuristicAgent(rules, bid_offset=k)
        try:
            report = evaluate_pairs(rules, space, seeds, dealers, agent, opponent, n_resamples=500)
        except RuntimeError:  # every pair aborted: nobody ever bid
            continue
        profile[k] = report.margin.mean
    return profile


def train(config: TrainConfig, rules: RuleConfig | None = None) -> None:
    rules = rules if rules is not None else load_default()
    space = build_action_space(rules)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    critic: CentralCritic | None = None
    if config.init_checkpoint is not None:
        net, payload = load_checkpoint(config.init_checkpoint, rules)
        net.train()
        if config.central_critic:
            critic = critic_from_payload(payload)
    else:
        net = build_network(rules, hidden=config.hidden)
        apply_action_prior(
            net,
            raise_decay=config.raise_prior_decay,
            capot_bias=config.capot_prior_bias,
            double_bias=config.double_prior_bias,
        )
    if config.central_critic and critic is None:
        critic = CentralCritic(hidden=config.critic_hidden)
    if critic is not None:
        critic.train()
    params = list(net.parameters()) + (list(critic.parameters()) if critic is not None else [])
    optimizer = torch.optim.Adam(params, lr=config.ppo.lr)
    warmup_params = critic.parameters() if critic is not None else net.value_head.parameters()
    warmup_optimizer = torch.optim.Adam(warmup_params, lr=1e-3)
    pool = OpponentPool(max_size=config.pool_max_size)
    writer = SummaryWriter(log_dir=str(config.log_dir))

    eval_rng = random.Random(config.seed + 1)
    random_agent = RandomAgent(random.Random(config.seed + 2))
    heuristic_agent = HeuristicAgent(rules, random.Random(config.seed + 3))
    # Fixed baselines the opponent pool always contains, from iteration 0 -- not just historical
    # self-play snapshots, which only start existing after `snapshot_every` iterations. See
    # `learn/rollout.py::collect_rollout`'s docstring: pure self-play among four identical,
    # still-randomly-initialized seats can collapse to a degenerate "everyone always passes"
    # equilibrium within a handful of updates, and a fixed baseline that reliably contests
    # auctions is what breaks that symmetry.
    passer_agent: Agent = ComposedAgent(PassBidder(), heuristic_agent)
    baseline_opponents: list[Opponent] = [
        _build_opponent(name, rules, config, heuristic_agent, random_agent, passer_agent)
        for name in config.baseline_opponents
    ]
    if config.bid_agent not in (None, "heuristic"):
        raise ValueError(f"unknown bid_agent {config.bid_agent!r}; expected None or 'heuristic'")
    bid_agent = heuristic_agent if config.bid_agent == "heuristic" else None

    best_score = float("-inf")
    for iteration in range(config.n_iterations):
        forced_bid_prob = _forced_bid_prob_at(iteration, config)
        forced_contra_prob = _forced_contra_prob_at(iteration, config)
        t0 = time.perf_counter()
        transitions, stats = collect_rollout(
            rules,
            space,
            net,
            n_deals=config.n_deals_per_iter,
            seed=config.seed * 1_000_003 + iteration,
            opponent_pool=[*baseline_opponents, *pool.snapshots],
            opponent_prob=config.opponent_prob,
            gamma=config.gamma,
            lam=config.lam,
            forced_bid_prob=forced_bid_prob,
            bid_agent=bid_agent,
            critic=critic,
            forced_contra_prob=forced_contra_prob,
            reward_scale=config.reward_scale,
        )
        t_rollout = time.perf_counter() - t0

        entropy_coef = _entropy_coef_at(iteration, config)
        ppo_config = replace(config.ppo, entropy_coef=entropy_coef)
        step_optimizer = optimizer
        if iteration < config.value_warmup_iters:
            ppo_config = replace(
                config.ppo,
                policy_coef=0.0,
                entropy_coef=0.0,
                belief_coef=0.0,
                outcome_coef=0.0,
                target_kl=None,
            )
            step_optimizer = warmup_optimizer
        t0 = time.perf_counter()
        update_stats = ppo_update(
            net, space, transitions, step_optimizer, ppo_config, critic=critic
        )
        t_update = time.perf_counter() - t0

        margin_mean = float(np.mean(stats.raw_margins)) if stats.raw_margins else 0.0
        abort_rate = stats.n_aborted / max(1, stats.n_deals)
        declare_rate = stats.learner_declare_rate
        writer.add_scalar("rollout/n_transitions", stats.n_transitions, iteration)
        writer.add_scalar("rollout/n_aborted", stats.n_aborted, iteration)
        writer.add_scalar("rollout/abort_rate", abort_rate, iteration)
        writer.add_scalar("rollout/margin_mean", margin_mean, iteration)
        if declare_rate is not None:
            writer.add_scalar("rollout/learner_declare_rate", declare_rate, iteration)
        writer.add_scalar("ppo/entropy_coef", entropy_coef, iteration)
        writer.add_scalar("rollout/forced_bid_prob", forced_bid_prob, iteration)
        writer.add_scalar("ppo/policy_loss", update_stats.policy_loss, iteration)
        writer.add_scalar("ppo/value_loss", update_stats.value_loss, iteration)
        writer.add_scalar("ppo/entropy", update_stats.entropy, iteration)
        writer.add_scalar("ppo/approx_kl", update_stats.approx_kl, iteration)
        writer.add_scalar("ppo/clip_fraction", update_stats.clip_fraction, iteration)
        if config.ppo.belief_coef > 0.0:
            writer.add_scalar("ppo/belief_loss", update_stats.belief_loss, iteration)
            writer.add_scalar("ppo/belief_acc", update_stats.belief_acc, iteration)
        if config.ppo.outcome_coef > 0.0:
            writer.add_scalar("ppo/outcome_loss", update_stats.outcome_loss, iteration)
            writer.add_scalar("ppo/outcome_acc", update_stats.outcome_acc, iteration)
        writer.add_scalar("rollout/n_capped", stats.n_capped, iteration)
        sp_n = len(stats.selfplay_levels)
        sp_str = "n/a"
        if sp_n:
            sp_level = float(np.median(stats.selfplay_levels))
            sp_made = stats.n_selfplay_made / sp_n
            sp_dbl = stats.n_selfplay_doubled / sp_n
            writer.add_scalar("selfplay/median_level", sp_level, iteration)
            writer.add_scalar("selfplay/made_rate", sp_made, iteration)
            writer.add_scalar("selfplay/doubled_rate", sp_dbl, iteration)
            sp_str = f"lvl={sp_level:.0f} made={sp_made:.2f} dbl={sp_dbl:.2f}"
        writer.add_scalar("time/rollout_s", t_rollout, iteration)
        writer.add_scalar("time/update_s", t_update, iteration)

        double_chances = [
            t
            for t in transitions
            if t.decision.phase == Phase.AUCTION and (t.decision.mask[1] or t.decision.mask[2])
        ]
        doubles = [t for t in double_chances if t.decision.action_idx in (1, 2)]
        policy_doubles = sum(not t.decision.forced for t in doubles)
        writer.add_scalar("rollout/policy_doubles", policy_doubles, iteration)
        writer.add_scalar("rollout/forced_doubles", len(doubles) - policy_doubles, iteration)
        writer.add_scalar("rollout/double_chances", len(double_chances), iteration)
        declare_str = f"{declare_rate:.2f}" if declare_rate is not None else "n/a"
        print(
            f"iter {iteration:5d}  margin={margin_mean:+7.2f}  abort_rate={abort_rate:.2f}  "
            f"declare_rate={declare_str}  ent_coef={entropy_coef:.3f}  "
            f"fbid={forced_bid_prob:.2f}  fdbl={forced_contra_prob:.2f}  "
            f"doubles={policy_doubles}+{len(doubles) - policy_doubles}f/{len(double_chances)}  "
            f"value_loss={update_stats.value_loss:.4f}  entropy={update_stats.entropy:.3f}  "
            f"kl={update_stats.approx_kl:+.4f}  clip={update_stats.clip_fraction:.3f}  "
            f"belief_acc={update_stats.belief_acc:.3f}  "
            f"outcome_acc={update_stats.outcome_acc:.3f}  selfplay[{sp_str}]  "
            f"capped={stats.n_capped}  "
            f"t=({t_rollout:.1f}s rollout, {t_update:.1f}s update)"
        )

        if (iteration + 1) % config.snapshot_every == 0:
            pool.add(net)

        if (iteration + 1) % config.eval_every == 0:
            net.eval()
            nn_agent: Agent = NNAgent(net, sample=False)
            random_baseline: Agent = random_agent
            if bid_agent is not None:
                nn_agent = ComposedAgent(bid_agent, nn_agent)
                random_baseline = ComposedAgent(bid_agent, random_agent)
            seeds = [eval_rng.randrange(1 << 30) for _ in range(config.eval_n_pairs)]
            dealers = [eval_rng.randrange(4) for _ in range(config.eval_n_pairs)]
            report_vs_random = evaluate_pairs(
                rules, space, seeds, dealers, nn_agent, random_baseline, n_resamples=2000
            )
            report_vs_heuristic = evaluate_pairs(
                rules, space, seeds, dealers, nn_agent, heuristic_agent, n_resamples=2000
            )
            if bid_agent is None:
                try:
                    report_vs_passer = evaluate_pairs(
                        rules,
                        space,
                        seeds,
                        dealers,
                        nn_agent,
                        passer_agent,
                        n_resamples=2000,
                    )
                except RuntimeError:
                    # Every pair aborted: neither the agent nor the always-pass bidder ever opened.
                    # That IS the collapse this comparison exists to catch, so report it rather
                    # than letting the evaluation crash the run.
                    print("  [eval:vs_passer] every pair aborted: the agent never bid")
                    writer.add_scalar("eval/never_bids_vs_passer", 1.0, iteration)
                else:
                    _log_eval(writer, "vs_passer", report_vs_passer, iteration)
            behavior = evaluate_behavior(
                rules, space, nn_agent, heuristic_agent, seeds[:60], dealers[:60]
            )
            print(f"  [behavior vs heuristic] {behavior.summary()}")
            writer.add_scalar("behavior/contract_rate", behavior.contract_rate, iteration)
            writer.add_scalar(
                "behavior/selfplay_contract_rate", behavior.selfplay_contract_rate, iteration
            )
            if behavior.declare_rate is not None:
                writer.add_scalar("behavior/declare_rate", behavior.declare_rate, iteration)
            if behavior.made_rate is not None:
                writer.add_scalar("behavior/made_rate", behavior.made_rate, iteration)
            if behavior.contra_rate is not None:
                writer.add_scalar("behavior/contra_rate", behavior.contra_rate, iteration)
            if bid_agent is None and config.signaling_deals > 0:
                signaling = measure_signaling(
                    rules, space, net, n_deals=config.signaling_deals, seed=config.seed + 7
                )
                if signaling is not None:
                    print(f"  [signaling] {signaling.summary()}")
                    writer.add_scalar(
                        "signaling/partner_info_gain", signaling.partner_info_gain.mean, iteration
                    )
                    writer.add_scalar(
                        "signaling/card_change_rate", signaling.card_change_rate.mean, iteration
                    )
            _log_eval(writer, "vs_random", report_vs_random, iteration)
            _log_eval(writer, "vs_heuristic", report_vs_heuristic, iteration)
            if config.heldout_offsets:
                profile = _heldout_profile(
                    rules,
                    space,
                    nn_agent,
                    config.heldout_offsets,
                    seeds[: config.heldout_pairs],
                    dealers[: config.heldout_pairs],
                )
                if profile:
                    # The score also counts the in-pool heuristic, so "best" cannot be won by
                    # trading it away for the held-out styles.
                    score = float(np.mean([*profile.values(), report_vs_heuristic.margin.mean]))
                    cells = "  ".join(f"{k:+d}:{m:+.1f}" for k, m in sorted(profile.items()))
                    print(
                        f"  [profile vs heuristic bid_offset] {cells}  | mean "
                        f"{float(np.mean(list(profile.values()))):+.2f}, min "
                        f"{min(profile.values()):+.1f}, score {score:+.2f}"
                    )
                    writer.add_scalar(
                        "profile/mean", float(np.mean(list(profile.values()))), iteration
                    )
                    writer.add_scalar("profile/min", min(profile.values()), iteration)
                    if score > best_score:
                        best_score = score
                        save_checkpoint(
                            config.checkpoint_dir / "best_profile.pt",
                            net,
                            rules,
                            extra={"iteration": iteration + 1, "profile_score": score},
                            critic=critic,
                        )
            net.train()

            save_checkpoint(
                config.checkpoint_dir / f"iter_{iteration + 1:06d}.pt",
                net,
                rules,
                extra={"iteration": iteration + 1},
                critic=critic,
            )

    save_checkpoint(
        config.checkpoint_dir / "final.pt",
        net,
        rules,
        extra={"iteration": config.n_iterations},
        critic=critic,
    )
    writer.close()
