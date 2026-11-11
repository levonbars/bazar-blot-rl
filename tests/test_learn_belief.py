"""M7: belief-head labels, loss and accuracy. The labels come from the ground-truth deal, so the
property that matters is that they describe exactly the cards the acting seat CANNOT already see
(other seats' unplayed hands) and nothing else."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from bazarblot.core.deal import Deal  # noqa: E402
from bazarblot.core.dealing import deal_hands  # noqa: E402
from bazarblot.core.rules import load_default  # noqa: E402
from bazarblot.learn.belief import (  # noqa: E402
    IGNORE,
    belief_accuracy,
    belief_loss,
    belief_target,
)

RULES = load_default()


def _fresh_hands() -> list[frozenset[int]]:
    deal = Deal(RULES, dealer=0, hands=deal_hands(random.Random(0), RULES), deal_id=0)
    return deal.hands


def test_belief_target_labels_exactly_the_other_seats_cards() -> None:
    hands = _fresh_hands()
    for seat in range(4):
        target = belief_target(hands, seat)
        assert int((target != IGNORE).sum()) == 24  # 3 other seats x 8 cards
        for card in hands[seat]:
            assert target[card] == IGNORE  # own cards are visible, nothing to predict
        for other in range(4):
            if other == seat:
                continue
            for card in hands[other]:
                assert target[card] == (other - seat) % 4 - 1


def test_belief_target_drops_cards_once_played() -> None:
    hands = _fresh_hands()
    played = next(iter(hands[1]))
    after = [hands[0], hands[1] - {played}, hands[2], hands[3]]
    assert belief_target(hands, 0)[played] == 0  # seat 1 is rel 1 -> index 0
    assert belief_target(after, 0)[played] == IGNORE


def test_belief_loss_ignores_unlabelled_cards_and_is_zero_with_no_labels() -> None:
    logits = torch.zeros(2, 32, 3, requires_grad=True)
    all_ignored = torch.full((2, 32), IGNORE, dtype=torch.long)
    loss = belief_loss(logits, all_ignored)
    assert loss.item() == 0.0
    loss.backward()  # graph intact, no error

    targets = torch.full((2, 32), IGNORE, dtype=torch.long)
    targets[0, 0] = 2
    uniform = belief_loss(torch.zeros(2, 32, 3), targets)
    assert uniform.item() == pytest.approx(torch.log(torch.tensor(3.0)).item(), abs=1e-5)


def test_belief_accuracy_counts_only_labelled_cards() -> None:
    logits = torch.zeros(1, 32, 3)
    logits[0, 0, 1] = 5.0  # predicts class 1
    logits[0, 1, 2] = 5.0  # predicts class 2
    targets = torch.full((1, 32), IGNORE, dtype=torch.long)
    targets[0, 0] = 1  # correct
    targets[0, 1] = 0  # wrong
    assert belief_accuracy(logits, targets) == pytest.approx(0.5)
