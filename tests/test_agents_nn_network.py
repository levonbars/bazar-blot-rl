"""M7: the policy/value network's shapes and basic determinism -- not its training behavior
(that's `tests/test_learn_ppo.py`). Skipped entirely if the `learn` extra (torch) isn't
installed, matching the project-wide rule that `core`/`env`/non-`nn` `agents` stay torch-free and
their own tests must keep passing without it."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bazarblot.agents.nn.network import OBS_DIM, NetworkHeads, build_network  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.env.actions import build_action_space  # noqa: E402
from bazarblot.env.obs import flatten  # noqa: E402

RULES = load_default()
SPACE = build_action_space(RULES)


def test_obs_dim_matches_flatten_output() -> None:
    import random

    from bazarblot.core.deal import Deal
    from bazarblot.core.dealing import deal_hands
    from bazarblot.env.infoset import info_set
    from bazarblot.env.obs import encode
    from bazarblot.env.tracked_deal import TrackedDeal

    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(0), RULES), deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, 0, match_score=(0, 0), deal_number=0)
    assert flatten(encode(info)).shape == (OBS_DIM,)


def test_network_head_widths_match_action_space() -> None:
    heads = NetworkHeads.from_space(SPACE)
    net = build_network(RULES)
    assert net.delta_head.out_features == heads.n_delta
    assert net.type_head.out_features == heads.n_types == 5
    assert net.capot_head.out_features == heads.capot_states == 2
    assert net.card_head.out_features == heads.n_cards == 32


def test_forward_shapes_and_determinism() -> None:
    torch.manual_seed(0)
    net = build_network(RULES)
    net.eval()
    obs = torch.zeros(3, OBS_DIM)
    out1 = net(obs)
    out2 = net(obs)
    assert out1["value"].shape == (3,)
    assert out1["meta"].shape == (3, 4)
    assert out1["delta"].shape == (3, NetworkHeads.from_space(SPACE).n_delta)
    assert out1["type"].shape == (3, 5)
    assert out1["capot"].shape == (3, 2)
    assert out1["card"].shape == (3, 32)
    for key in out1:
        assert torch.equal(out1[key], out2[key])
