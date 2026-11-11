"""M7 step 3: the centralized critic (a training-only value function that sees the hidden hands).
The properties that matter: it is built from exactly the information the belief labels carry, the
rollout uses it as the advantage baseline, only the critic (not the actor) is trained by its value
loss, and the actor's input never changes."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.heuristic import HeuristicAgent  # noqa: E402
from bazarblot.agents.nn.network import (  # noqa: E402
    OBS_DIM,
    PRIVILEGED_DIM,
    CentralCritic,
    build_network,
)
from bazarblot.core.deal import Phase  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space  # noqa: E402
from bazarblot.learn.belief import IGNORE, privileged_features  # noqa: E402
from bazarblot.learn.checkpoint import (  # noqa: E402
    critic_from_payload,
    load_checkpoint,
    save_checkpoint,
)
from bazarblot.learn.ppo import PPOConfig, _advantages, ppo_update  # noqa: E402
from bazarblot.learn.rollout import collect_rollout  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def test_privileged_features_are_a_one_hot_of_the_holder_labels() -> None:
    targets = np.full((2, 32), IGNORE, dtype=np.int8)
    targets[0, 0] = 2
    targets[0, 5] = 0
    targets[1, 31] = 1
    feats = privileged_features(targets)
    assert feats.shape == (2, PRIVILEGED_DIM)
    assert feats[0, 0 * 3 + 2] == 1.0 and feats[0, 5 * 3 + 0] == 1.0 and feats[1, 31 * 3 + 1] == 1.0
    assert feats.sum() == 3.0  # exactly the labelled cards, nothing else
    assert privileged_features(np.full((32,), IGNORE, dtype=np.int8)).sum() == 0.0


def test_critic_output_depends_on_the_hidden_hands() -> None:
    torch.manual_seed(0)
    critic = CentralCritic()
    obs = torch.zeros(1, OBS_DIM)
    a = critic(obs, torch.zeros(1, PRIVILEGED_DIM))
    b = critic(obs, torch.ones(1, PRIVILEGED_DIM))
    assert a.shape == (1,)
    assert not torch.equal(a, b)


def test_rollout_uses_the_critic_as_the_baseline() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    critic = CentralCritic()
    transitions, _ = collect_rollout(RULES, SPACE, net, n_deals=12, seed=0, critic=critic)
    assert transitions
    # Baselines now come from the critic, not the actor's value head...
    assert any(abs(t.value - t.decision.value) > 1e-6 for t in transitions)
    # ...and the advantage is consistent with them: return = advantage + baseline.
    for t in transitions:
        assert t.ret == pytest.approx(t.advantage + t.value, abs=1e-5)


def test_critic_value_loss_trains_the_critic_and_leaves_the_actor_value_head_alone() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    critic = CentralCritic()
    transitions, _ = collect_rollout(RULES, SPACE, net, n_deals=15, seed=0, critic=critic)
    net_before = {k: v.clone() for k, v in net.state_dict().items()}
    critic_before = {k: v.clone() for k, v in critic.state_dict().items()}
    # Critic-only optimizer, policy/entropy/belief off: only the critic may move.
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    config = PPOConfig(
        epochs=2,
        minibatch_size=64,
        policy_coef=0.0,
        entropy_coef=0.0,
        belief_coef=0.0,
        target_kl=None,
    )
    ppo_update(net, SPACE, transitions, optimizer, config, critic=critic)
    assert all(torch.equal(net_before[k], v) for k, v in net.state_dict().items())
    assert any(not torch.equal(critic_before[k], v) for k, v in critic.state_dict().items())


def test_actor_input_does_not_change_when_a_critic_is_used() -> None:
    """The actor's observation must be identical with and without a critic: hidden hands may only
    ever feed the critic."""
    torch.manual_seed(0)
    net = build_network(RULES)
    critic = (
        CentralCritic()
    )  # built before the rollouts: its init must not consume their randomness
    torch.manual_seed(1)
    plain, _ = collect_rollout(RULES, SPACE, net, n_deals=8, seed=3)
    torch.manual_seed(1)
    with_critic, _ = collect_rollout(RULES, SPACE, net, n_deals=8, seed=3, critic=critic)
    assert len(plain) == len(with_critic)
    for p, c in zip(plain, with_critic, strict=True):
        assert np.array_equal(p.decision.obs, c.decision.obs)
        assert p.decision.action_idx == c.decision.action_idx


def test_per_phase_normalization_equalizes_auction_and_play_scales() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    transitions, _ = collect_rollout(RULES, SPACE, net, n_deals=30, seed=0)
    for t in transitions:  # give the auction decisions a 20x larger advantage scale
        if t.decision.phase == Phase.AUCTION:
            t.advantage *= 20.0
    auction = np.array([t.decision.phase == Phase.AUCTION for t in transitions])
    per_phase = _advantages(transitions, per_phase=True)
    global_ = _advantages(transitions, per_phase=False)
    assert per_phase[auction].std() == pytest.approx(1.0, abs=1e-3)
    assert per_phase[~auction].std() == pytest.approx(1.0, abs=1e-3)
    assert global_[auction].std() > 5 * global_[~auction].std()  # the swamping it prevents


def test_checkpoint_round_trips_the_critic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    net, critic = build_network(RULES), CentralCritic()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, RULES, extra={"iteration": 1}, critic=critic)
    _, payload = load_checkpoint(path, RULES)
    restored = critic_from_payload(payload)
    assert restored is not None
    for k, v in critic.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k])
    save_checkpoint(path, net, RULES, extra={"iteration": 1})
    assert critic_from_payload(load_checkpoint(path, RULES)[1]) is None


def test_heuristic_opponent_still_works_with_a_critic() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    transitions, stats = collect_rollout(
        RULES,
        SPACE,
        net,
        n_deals=10,
        seed=0,
        opponent_pool=[HeuristicAgent(RULES)],
        opponent_prob=1.0,
        critic=CentralCritic(),
    )
    assert stats.n_opponent_deals == 10 and transitions
