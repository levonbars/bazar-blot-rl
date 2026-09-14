"""The PPO clipped-surrogate update (M7), operating over a flat batch of `Transition`s collected
by `learn/rollout.py`. Each transition's log-probability is recomputed under the CURRENT network
parameters via `agents/nn/ppo_agent._recompute_log_prob_from_output` -- the same hierarchical
decomposition `decide()` used to sample it originally, scoring the SAME action -- so the
importance ratio `exp(new_log_prob - old_log_prob)` is exact for the hybrid meta/bid/card action
structure, with no special-casing needed at update time.

**The network's forward pass is batched once per minibatch**, not once per transition: earlier
versions called `recompute_log_prob` (one `net()` call per transition) in a Python loop, which
profiling of a real training run showed dominating update time once minibatches reached a few
hundred transitions. AUCTION and PLAY transitions need different head subsets and different mask
shapes, but the network itself doesn't care about phase -- it always computes every head -- so one
`net(obs_batch)` call produces every sample's logits, and only the (cheap, no-autograd-call)
per-sample hierarchical bookkeeping stays in a Python loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import optim

from bazarblot.agents.nn.network import PolicyValueNet
from bazarblot.agents.nn.ppo_agent import _recompute_log_prob_from_output
from bazarblot.env.actions import ActionSpace
from bazarblot.learn.rollout import Transition


@dataclass(frozen=True, slots=True)
class PPOConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    """Stop taking further epochs over this batch once a minibatch's `approx_kl` exceeds this --
    standard PPO early-stopping (e.g. Stable-Baselines3's default `target_kl`). Observed as
    necessary during M7: `approx_kl` reached 1-6 (vs. a healthy ~0.01-0.02) once forced-bid
    exploration (`learn/rollout.py`'s `forced_bid_prob`) started regularly injecting actions the
    network's own distribution assigned low probability to -- `clip_ratio` alone bounds the
    surrogate objective's gradient per sample, but not how far several epochs of updates can move
    the policy in total. `None` disables the check (the pre-M7-instability-fix behavior)."""


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


def _normalize(advantages: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    std = advantages.std()
    centered = advantages - advantages.mean()
    result = centered if std < 1e-8 else centered / std
    return np.asarray(result, dtype=np.float32)


def ppo_update(
    net: PolicyValueNet,
    space: ActionSpace,
    transitions: list[Transition],
    optimizer: optim.Optimizer,
    config: PPOConfig,
) -> PPOUpdateStats:
    """One full PPO pass (`config.epochs` epochs over `transitions`, shuffled into
    `config.minibatch_size` minibatches each epoch). Advantages are normalized once, over the
    whole batch, before any epoch -- standard PPO practice, and important here specifically
    because per-seat trajectory lengths vary (auction-only vs. full-play decisions), so raw
    advantage scale would otherwise differ by trajectory position in a way normalization removes.

    **`transitions` can legitimately be empty**: `learn/rollout.py` discards every deal that
    aborts (4 passes, no bid), and a rollout batch where every single shuffle aborts -- rare, but
    not astronomically so under a policy still exploring heavily early in training -- would
    otherwise divide by zero below. Skip the update entirely and report all-zero stats rather than
    crash a training run over one unlucky (or one badly-behaved-so-far) batch.
    """
    if not transitions:
        return PPOUpdateStats(
            policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0, clip_fraction=0.0
        )

    advantages = _normalize(np.array([t.advantage for t in transitions], dtype=np.float32))
    returns = np.array([t.ret for t in transitions], dtype=np.float32)
    old_log_probs = np.array([t.decision.log_prob for t in transitions], dtype=np.float32)

    n = len(transitions)
    stats_accum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    n_clipped = 0
    n_updates = 0

    stop_early = False
    for _epoch in range(config.epochs):
        if stop_early:
            break
        order = np.random.permutation(n)
        for start in range(0, n, config.minibatch_size):
            idx = order[start : start + config.minibatch_size]
            obs_batch = torch.from_numpy(np.stack([transitions[i].decision.obs for i in idx]))
            out_batch = net(obs_batch)

            batch_log_probs = []
            batch_entropies = []
            batch_values = []
            for j, i in enumerate(idx):
                t = transitions[i]
                out_i = {k: v[j] for k, v in out_batch.items()}
                log_prob, entropy, value = _recompute_log_prob_from_output(
                    space,
                    out_i,
                    t.decision.mask,
                    t.decision.phase,
                    t.decision.base_level,
                    t.decision.action_idx,
                )
                batch_log_probs.append(log_prob)
                batch_entropies.append(entropy)
                batch_values.append(value)

            new_log_probs = torch.stack(batch_log_probs)
            entropies = torch.stack(batch_entropies)
            values = torch.stack(batch_values)

            batch_advantages = torch.from_numpy(advantages[idx])
            batch_returns = torch.from_numpy(returns[idx])
            batch_old_log_probs = torch.from_numpy(old_log_probs[idx])

            with torch.no_grad():
                approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
            if config.target_kl is not None and approx_kl > config.target_kl:
                # The policy has already drifted too far from the rollout-time policy (from
                # earlier minibatches/epochs on this same batch) -- stop taking further gradient
                # steps on it rather than risk the same instability that motivated adding this
                # check (see `PPOConfig.target_kl`'s docstring). This minibatch's own step is
                # skipped too, matching standard practice (Stable-Baselines3's `target_kl`).
                stop_early = True
                break

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
            clipped = clipped * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch_returns)
            entropy_bonus = entropies.mean()

            loss = (
                policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                n_clipped += int((torch.abs(ratio - 1.0) > config.clip_ratio).sum().item())

            stats_accum["policy_loss"] += float(policy_loss.item())
            stats_accum["value_loss"] += float(value_loss.item())
            stats_accum["entropy"] += float(entropy_bonus.item())
            stats_accum["approx_kl"] += approx_kl
            n_updates += 1

    if n_updates == 0:
        # `target_kl` tripped on the very first minibatch (old == new policy at that point, so
        # only possible with a pathologically tiny `target_kl`) -- no gradient step happened at
        # all this call; report zeroed stats rather than divide by zero, same reasoning as the
        # empty-`transitions` case above.
        return PPOUpdateStats(
            policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0, clip_fraction=0.0
        )

    return PPOUpdateStats(
        policy_loss=stats_accum["policy_loss"] / n_updates,
        value_loss=stats_accum["value_loss"] / n_updates,
        entropy=stats_accum["entropy"] / n_updates,
        approx_kl=stats_accum["approx_kl"] / n_updates,
        clip_fraction=n_clipped / max(1, n * config.epochs),
    )
