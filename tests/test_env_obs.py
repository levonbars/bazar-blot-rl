"""M3: the observation encoder. Shapes, determinism, the fixed-total-dimension property Gym/
PettingZoo require, and a spot-check of a few blocks' semantics against hand-computable values."""

from __future__ import annotations

import random

import numpy as np

from bazarblot.core.deal import Deal, Phase
from bazarblot.core.dealing import deal_hands
from bazarblot.core.rules import load_default
from bazarblot.env.actions import build_action_space, legal_mask
from bazarblot.env.infoset import info_set
from bazarblot.env.obs import BLOCK_SHAPES, OBS_VERSION, encode, flatten
from bazarblot.env.tracked_deal import TrackedDeal

RULES = load_default()
SPACE = build_action_space(RULES)
_FLAT_DIM = sum(int(np.prod(s)) for s in BLOCK_SHAPES.values())


def test_obs_version_is_stamped() -> None:
    assert OBS_VERSION == "v1"


def _play_random_deal(seed: int) -> tuple[Deal, TrackedDeal]:
    rng = random.Random(seed)
    hands = deal_hands(rng, RULES)
    deal = Deal(RULES, dealer=seed % 4, hands=hands, deal_id=seed)
    tracked = TrackedDeal(deal)
    steps = 0
    while deal.phase in (Phase.AUCTION, Phase.PLAY) and steps < 500:
        seat = deal.to_act
        info = info_set(tracked, seat, match_score=(3, 5), deal_number=2)
        mask = legal_mask(info, SPACE)
        legal = mask.nonzero()[0]
        idx = rng.choice(list(legal))
        action = SPACE.decode(int(idx))
        tracked.step(action)
        steps += 1
    return deal, tracked


def test_every_block_has_declared_shape_and_is_finite() -> None:
    for seed in range(80):
        _deal, tracked = _play_random_deal(seed)
        for seat in range(4):
            info = info_set(tracked, seat, match_score=(3, 5), deal_number=2)
            blocks = encode(info)
            assert set(blocks) == set(BLOCK_SHAPES)
            for key, shape in BLOCK_SHAPES.items():
                assert blocks[key].shape == shape, (seed, seat, key)
                assert blocks[key].dtype == np.float32
                assert np.isfinite(blocks[key]).all(), (seed, seat, key)


def test_flatten_is_fixed_dimension_regardless_of_phase() -> None:
    shapes_seen = set()
    for seed in range(50):
        _deal, tracked = _play_random_deal(seed)
        for seat in range(4):
            info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
            flat = flatten(encode(info))
            shapes_seen.add(flat.shape)
            assert flat.dtype == np.float32
    assert shapes_seen == {(_FLAT_DIM,)}


def test_encode_is_deterministic() -> None:
    """Same InfoSet -> bit-identical encoding, every time. The leakage test in
    `test_env_infoset.py` depends on this holding."""
    _deal, tracked = _play_random_deal(7)
    info = info_set(tracked, seat=0, match_score=(1, 2), deal_number=1)
    a = flatten(encode(info))
    b = flatten(encode(info))
    assert np.array_equal(a, b)


def test_hand_block_matches_infoset_hand() -> None:
    _deal, tracked = _play_random_deal(3)
    for seat in range(4):
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        blocks = encode(info)
        for c in range(32):
            assert blocks["hand"][c] == float(c in info.hand)


def test_cards_left_matches_hand_sizes() -> None:
    _deal, tracked = _play_random_deal(11)
    n = RULES.deal.cards_per_player
    for seat in range(4):
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        blocks = encode(info)
        for other in range(4):
            rel = (other - seat) % 4
            assert blocks["cards_left"][rel] == info.hand_sizes[other] / n


def test_trick_index_matches_completed_trick_count() -> None:
    _deal, tracked = _play_random_deal(21)
    for seat in range(4):
        info = info_set(tracked, seat, match_score=(0, 0), deal_number=0)
        blocks = encode(info)
        expected = min(len(info.tricks), 7)
        assert blocks["trick_index"][expected] == 1.0
        assert blocks["trick_index"].sum() == 1.0


def test_melds_public_none_before_contract_all_zero() -> None:
    rng = random.Random(0)
    hands = deal_hands(rng, RULES)
    deal = Deal(RULES, dealer=0, hands=hands, deal_id=0)
    tracked = TrackedDeal(deal)
    info = info_set(tracked, 0, match_score=(0, 0), deal_number=0)
    assert info.phase == Phase.AUCTION
    blocks = encode(info)
    assert np.array_equal(blocks["melds_public"], np.zeros(BLOCK_SHAPES["melds_public"]))
