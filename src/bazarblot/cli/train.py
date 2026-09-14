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
        ppo=PPOConfig(lr=args.lr),
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        forced_bid_prob_start=args.forced_bid_prob_start,
        forced_bid_prob_end=args.forced_bid_prob_end,
        forced_bid_anneal_frac=args.forced_bid_anneal_frac,
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
