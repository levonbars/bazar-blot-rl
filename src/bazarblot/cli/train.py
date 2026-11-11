"""M7 entry point: `python -m bazarblot.cli.train [options]` runs self-play PPO training.

Run `python -m bazarblot.cli.train --help` for the full option list; defaults match
`learn.train.TrainConfig`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bazarblot.core.rules import load_default
from bazarblot.learn.ppo import PPOConfig
from bazarblot.learn.train import TrainConfig, train


def _parse_args(argv: list[str]) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--deals-per-iter", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-coef-start", type=float, default=0.05)
    p.add_argument("--entropy-coef-end", type=float, default=0.02)
    p.add_argument("--forced-bid-prob-start", type=float, default=0.4)
    p.add_argument("--forced-bid-prob-end", type=float, default=0.05)
    p.add_argument("--forced-bid-anneal-frac", type=float, default=1.0)
    p.add_argument("--reward-scale", type=float, default=24.0)
    p.add_argument("--forced-contra-prob-start", type=float, default=0.0)
    p.add_argument("--forced-contra-prob-end", type=float, default=0.0)
    p.add_argument(
        "--bid-agent",
        choices=["heuristic"],
        default=None,
        help="train card play only, with this agent making every auction decision",
    )
    p.add_argument("--belief-coef", type=float, default=0.0)
    p.add_argument("--outcome-coef", type=float, default=0.0)
    p.add_argument(
        "--raise-prior-decay",
        type=float,
        default=0.0,
        help="from-scratch start: lower the logit of a raise of k by decay*(k-1)",
    )
    p.add_argument("--capot-prior-bias", type=float, default=0.0)
    p.add_argument("--double-prior-bias", type=float, default=0.0)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--value-warmup-iters", type=int, default=0)
    p.add_argument(
        "--central-critic",
        action="store_true",
        help="train a centralized critic (sees hidden hands, training only) as the baseline",
    )
    p.add_argument(
        "--opponents",
        default="heuristic,random",
        help="comma-separated fixed training opponents: heuristic, heuristic@K (bid offset K), "
        "random, passer, clone",
    )
    p.add_argument(
        "--heldout-offsets",
        default="",
        help="comma-separated heuristic bid offsets to report a generalisation profile on "
        "(keep disjoint from the offsets in --opponents)",
    )
    p.add_argument("--heldout-pairs", type=int, default=100)
    p.add_argument("--opponent-prob", type=float, default=0.3)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-n-pairs", type=int, default=150)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/m7"))
    p.add_argument("--log-dir", type=Path, default=Path("runs/m7"))
    args = p.parse_args(argv[1:])

    return TrainConfig(
        n_iterations=args.iterations,
        n_deals_per_iter=args.deals_per_iter,
        seed=args.seed,
        ppo=PPOConfig(lr=args.lr, belief_coef=args.belief_coef, outcome_coef=args.outcome_coef),
        raise_prior_decay=args.raise_prior_decay,
        capot_prior_bias=args.capot_prior_bias,
        double_prior_bias=args.double_prior_bias,
        init_checkpoint=args.init_checkpoint,
        value_warmup_iters=args.value_warmup_iters,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        forced_bid_prob_start=args.forced_bid_prob_start,
        forced_bid_prob_end=args.forced_bid_prob_end,
        forced_bid_anneal_frac=args.forced_bid_anneal_frac,
        reward_scale=args.reward_scale,
        forced_contra_prob_start=args.forced_contra_prob_start,
        forced_contra_prob_end=args.forced_contra_prob_end,
        bid_agent=args.bid_agent,
        central_critic=args.central_critic,
        baseline_opponents=tuple(x for x in args.opponents.split(",") if x),
        heldout_offsets=tuple(int(x) for x in args.heldout_offsets.split(",") if x),
        heldout_pairs=args.heldout_pairs,
        opponent_prob=args.opponent_prob,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        eval_n_pairs=args.eval_n_pairs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )


def main(argv: list[str]) -> None:
    config = _parse_args(argv)
    train(config, rules=load_default())


if __name__ == "__main__":
    main(sys.argv)
