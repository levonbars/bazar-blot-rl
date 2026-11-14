"""Checkpoint save/load, versioned the same way `env/obs.py` versions observations: a checkpoint
records the `OBS_VERSION` and `rules_hash` it was trained under, and loading refuses a mismatch
rather than silently restoring weights whose input features no longer mean what the checkpoint
thinks they mean (spec §4's own rule for `OBS_VERSION`, applied here to the consumer side)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bazarblot.agents.nn.network import CentralCritic, PolicyValueNet, build_network
from bazarblot.core.rules import RuleConfig
from bazarblot.env.obs import OBS_VERSION


class CheckpointVersionMismatch(RuntimeError):
    pass


def save_checkpoint(
    path: Path,
    net: PolicyValueNet,
    rules: RuleConfig,
    extra: dict[str, Any],
    critic: CentralCritic | None = None,
) -> None:
    """Save the actor (and, if given, the training-only centralized critic, so a resumed run does
    not restart from an untrained baseline)."""
    payload = {
        "obs_version": OBS_VERSION,
        "rules_hash": rules.rules_hash,
        "hidden": _hidden_sizes(net),
        "state_dict": net.state_dict(),
        **extra,
    }
    if critic is not None:
        payload["critic_hidden"] = tuple(critic.hidden)
        payload["critic_state_dict"] = critic.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, rules: RuleConfig) -> tuple[PolicyValueNet, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["obs_version"] != OBS_VERSION:
        raise CheckpointVersionMismatch(
            f"checkpoint was trained under obs_version={payload['obs_version']!r}, "
            f"current code is {OBS_VERSION!r}"
        )
    if payload["rules_hash"] != rules.rules_hash:
        raise CheckpointVersionMismatch(
            f"checkpoint was trained under rules_hash={payload['rules_hash']!r}, "
            f"current preset is {rules.rules_hash!r}"
        )
    net = build_network(rules, hidden=tuple(payload["hidden"]))
    load_actor_state(net, payload["state_dict"])
    return net, payload


def load_actor_state(net: PolicyValueNet, state: dict[str, Any]) -> None:
    """Load actor weights; checkpoints from before the oracle input (`ORACLE_DIM`) lack only its
    zero-initialized weights, which keep their zeros (the network then behaves exactly as saved)."""
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or any(k != "oracle_weight" for k in missing):
        raise RuntimeError(f"checkpoint does not match the network: {missing=} {unexpected=}")


def _hidden_sizes(net: PolicyValueNet) -> tuple[int, ...]:
    return tuple(layer.out_features for layer in net.trunk if isinstance(layer, torch.nn.Linear))


def critic_from_payload(payload: dict[str, Any]) -> CentralCritic | None:
    """The centralized critic stored in a checkpoint payload, or `None` if it has none."""
    if "critic_state_dict" not in payload:
        return None
    critic = CentralCritic(hidden=tuple(payload["critic_hidden"]))
    critic.load_state_dict(payload["critic_state_dict"])
    return critic
