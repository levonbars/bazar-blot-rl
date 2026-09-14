"""The PPO clipped-surrogate update (M7), operating over a flat batch of `Transition`s collected
by `learn/rollout.py`. Each transition's log-probability is recomputed under the CURRENT network
parameters via `agents/nn/ppo_agent.recompute_log_prob` -- the same hierarchical decomposition
`decide()` used to sample it originally, scoring the SAME action -- so the importance ratio
`exp(new_log_prob - old_log_prob)` is exact for the hybrid meta/bid/card action structure, with no
special-casing needed at update time.

**Known scalability limit, not yet addressed:** `recompute_log_prob` runs one transition at a
time (batch size 1 per forward pass) rather than batching a whole minibatch through the network
at once, because AUCTION and PLAY transitions need different head subsets and different mask
shapes. Correct and simple; the obvious next step if update time becomes the training bottleneck
is to batch same-phase transitions together. Not done here -- premature before profiling the
actual training loop shows it matters, matching M6's own "measure before optimizing" discipline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import optim

from bazarblot.agents.nn.network import PolicyValueNet
from bazarblot.agents.nn.ppo_agent import recompute_log_prob
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

    for _epoch in range(config.epochs):
        order = np.random.permutation(n)
        for start in range(0, n, config.minibatch_size):
            idx = order[start : start + config.minibatch_size]
            batch_log_probs = []
            batch_entropies = []
            batch_values = []
            for i in idx:
                t = transitions[i]
                obs = torch.from_numpy(t.decision.obs)
                log_prob, entropy, value = recompute_log_prob(
                    net,
                    space,
                    obs,
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
                approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                n_clipped += int((torch.abs(ratio - 1.0) > config.clip_ratio).sum().item())

            stats_accum["policy_loss"] += float(policy_loss.item())
            stats_accum["value_loss"] += float(value_loss.item())
            stats_accum["entropy"] += float(entropy_bonus.item())
            stats_accum["approx_kl"] += approx_kl
            n_updates += 1

    return PPOUpdateStats(
        policy_loss=stats_accum["policy_loss"] / n_updates,
        value_loss=stats_accum["value_loss"] / n_updates,
        entropy=stats_accum["entropy"] / n_updates,
        approx_kl=stats_accum["approx_kl"] / n_updates,
        clip_fraction=n_clipped / max(1, n * config.epochs),
    )
