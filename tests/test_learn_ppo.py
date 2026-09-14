"""M7: the PPO update step. Two properties matter most: it runs end-to-end on real rollout data
without producing `nan`/`inf`, and repeated updates on a FIXED batch actually reduce the value
loss -- the cheapest possible sanity check that gradients are flowing to the right place and the
per-transition log-prob recomputation isn't silently detached from the graph."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.nn.network import build_network  # noqa: E402
from bazarblot.agents.nn.ppo_agent import recompute_log_prob  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space  # noqa: E402
from bazarblot.learn.ppo import PPOConfig, ppo_update  # noqa: E402
from bazarblot.learn.rollout import collect_rollout  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def test_ppo_update_runs_and_produces_finite_stats() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=20, seed=0)
    assert stats.n_transitions > 0

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    config = PPOConfig(epochs=2, minibatch_size=64)
    update_stats = ppo_update(net, SPACE, transitions, optimizer, config)

    for value in (
        update_stats.policy_loss,
        update_stats.value_loss,
        update_stats.entropy,
        update_stats.approx_kl,
    ):
        assert value == value  # not nan
        assert abs(value) < 1e6  # not inf / blown up


def test_ppo_update_on_empty_transitions_does_not_crash() -> None:
    """A rollout batch where every deal aborted is rare but real (see
    `learn/rollout.py`'s abort-discard policy) -- `ppo_update` must skip the update and return
    zeroed stats, not divide by zero. Regression test for a crash hit during the first real M7
    training run (see `docs/03-implementation-roadmap.md` M7)."""
    torch.manual_seed(0)
    net = build_network(RULES)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    stats = ppo_update(net, SPACE, [], optimizer, PPOConfig())
    assert stats.policy_loss == 0.0
    assert stats.value_loss == 0.0
    assert stats.clip_fraction == 0.0


def test_batched_forward_pass_matches_per_sample_recompute() -> None:
    """Regression test for the batching refactor (M7, `docs/03-implementation-roadmap.md`):
    `ppo_update` now runs ONE `net(obs_batch)` call per minibatch instead of one `net()` call per
    transition. Since `PolicyValueNet` has no batch-norm-like cross-sample dependencies (plain
    Linear/ReLU throughout), a batched forward pass must give bit-identical per-sample results to
    calling `recompute_log_prob` (the original, unbatched path) one transition at a time."""
    torch.manual_seed(0)
    net = build_network(RULES)
    net.eval()
    transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=20, seed=0)
    assert stats.n_transitions >= 8
    sample = transitions[:8]

    obs_batch = torch.stack([torch.from_numpy(t.decision.obs) for t in sample])
    with torch.no_grad():
        out_batch = net(obs_batch)

    from bazarblot.agents.nn.ppo_agent import _recompute_log_prob_from_output

    for j, t in enumerate(sample):
        out_i = {k: v[j] for k, v in out_batch.items()}
        batched_log_prob, batched_entropy, batched_value = _recompute_log_prob_from_output(
            SPACE,
            out_i,
            t.decision.mask,
            t.decision.phase,
            t.decision.base_level,
            t.decision.action_idx,
        )
        with torch.no_grad():
            unbatched_log_prob, unbatched_entropy, unbatched_value = recompute_log_prob(
                net,
                SPACE,
                torch.from_numpy(t.decision.obs),
                t.decision.mask,
                t.decision.phase,
                t.decision.base_level,
                t.decision.action_idx,
            )
        assert batched_log_prob.item() == pytest.approx(unbatched_log_prob.item(), abs=1e-6)
        assert batched_entropy.item() == pytest.approx(unbatched_entropy.item(), abs=1e-6)
        assert batched_value.item() == pytest.approx(unbatched_value.item(), abs=1e-6)


def test_target_kl_stops_updates_before_any_gradient_step() -> None:
    """An impossible-to-satisfy `target_kl` (below the ~0 KL of the very first minibatch) must
    stop `ppo_update` before it ever calls `optimizer.step()` -- verified by checking the
    network's parameters are byte-for-byte unchanged, not just that the reported stats are zero."""
    torch.manual_seed(0)
    net = build_network(RULES)
    transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=15, seed=0)
    assert stats.n_transitions > 0

    before = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    config = PPOConfig(epochs=3, minibatch_size=64, target_kl=-1.0)
    update_stats = ppo_update(net, SPACE, transitions, optimizer, config)

    assert update_stats.policy_loss == 0.0
    assert update_stats.value_loss == 0.0
    for key, value in net.state_dict().items():
        assert torch.equal(before[key], value), f"{key} changed despite target_kl stopping first"


def test_repeated_updates_on_a_fixed_batch_reduce_value_loss() -> None:
    torch.manual_seed(1)
    net = build_network(RULES)
    transitions, stats = collect_rollout(RULES, SPACE, net, n_deals=15, seed=1)
    assert stats.n_transitions > 0

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    config = PPOConfig(epochs=1, minibatch_size=len(transitions), clip_ratio=0.2)

    losses = [ppo_update(net, SPACE, transitions, optimizer, config).value_loss for _ in range(30)]

    # Noisy step-to-step (the policy loss and entropy bonus share the trunk and pull against a
    # pure value-regression objective), but comparing early-vs-late windows rather than single
    # first/last points, the value loss on this fixed batch should be decisively lower once the
    # optimizer has had 30 passes over it.
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 * 0.5
