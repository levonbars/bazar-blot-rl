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

from bazarblot.agents.heuristic import HeuristicAgent
from bazarblot.agents.nn.network import build_network
from bazarblot.agents.nn.ppo_agent import NNAgent
from bazarblot.agents.random_agent import RandomAgent
from bazarblot.core.rules import RuleConfig, load_default
from bazarblot.env.actions import build_action_space
from bazarblot.eval.metrics import evaluate_pairs
from bazarblot.learn.checkpoint import save_checkpoint
from bazarblot.learn.opponent_pool import OpponentPool
from bazarblot.learn.ppo import PPOConfig, ppo_update
from bazarblot.learn.rollout import Opponent, collect_rollout


@dataclass(frozen=True, slots=True)
class TrainConfig:
    n_iterations: int = 1000
    n_deals_per_iter: int = 256
    seed: int = 0
    hidden: tuple[int, ...] = (256, 256)
    gamma: float = 1.0
    lam: float = 0.95
    ppo: PPOConfig = field(default_factory=PPOConfig)
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.02
    forced_bid_prob_start: float = 0.4
    forced_bid_prob_end: float = 0.05
    forced_bid_anneal_frac: float = 1.0
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


def train(config: TrainConfig, rules: RuleConfig | None = None) -> None:
    rules = rules if rules is not None else load_default()
    space = build_action_space(rules)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    net = build_network(rules, hidden=config.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.ppo.lr)
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
    baseline_opponents: list[Opponent] = [heuristic_agent, random_agent]

    for iteration in range(config.n_iterations):
        forced_bid_prob = _forced_bid_prob_at(iteration, config)
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
        )
        t_rollout = time.perf_counter() - t0

        entropy_coef = _entropy_coef_at(iteration, config)
        ppo_config = replace(config.ppo, entropy_coef=entropy_coef)
        t0 = time.perf_counter()
        update_stats = ppo_update(net, space, transitions, optimizer, ppo_config)
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
        writer.add_scalar("time/rollout_s", t_rollout, iteration)
        writer.add_scalar("time/update_s", t_update, iteration)

        declare_str = f"{declare_rate:.2f}" if declare_rate is not None else "n/a"
        print(
            f"iter {iteration:5d}  margin={margin_mean:+7.2f}  abort_rate={abort_rate:.2f}  "
            f"declare_rate={declare_str}  ent_coef={entropy_coef:.3f}  "
            f"fbid={forced_bid_prob:.2f}  "
            f"value_loss={update_stats.value_loss:.4f}  entropy={update_stats.entropy:.3f}  "
            f"kl={update_stats.approx_kl:+.4f}  clip={update_stats.clip_fraction:.3f}  "
            f"t=({t_rollout:.1f}s rollout, {t_update:.1f}s update)"
        )

        if (iteration + 1) % config.snapshot_every == 0:
            pool.add(net)

        if (iteration + 1) % config.eval_every == 0:
            net.eval()
            nn_agent = NNAgent(net, sample=False)
            seeds = [eval_rng.randrange(1 << 30) for _ in range(config.eval_n_pairs)]
            dealers = [eval_rng.randrange(4) for _ in range(config.eval_n_pairs)]
            report_vs_random = evaluate_pairs(
                rules, space, seeds, dealers, nn_agent, random_agent, n_resamples=2000
            )
            report_vs_heuristic = evaluate_pairs(
                rules, space, seeds, dealers, nn_agent, heuristic_agent, n_resamples=2000
            )
            _log_eval(writer, "vs_random", report_vs_random, iteration)
            _log_eval(writer, "vs_heuristic", report_vs_heuristic, iteration)
            net.train()

            save_checkpoint(
                config.checkpoint_dir / f"iter_{iteration + 1:06d}.pt",
                net,
                rules,
                extra={"iteration": iteration + 1},
            )

    save_checkpoint(
        config.checkpoint_dir / "final.pt", net, rules, extra={"iteration": config.n_iterations}
    )
    writer.close()
