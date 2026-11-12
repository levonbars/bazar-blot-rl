"""The UI's choice of bots: simple, heuristic, or a trained checkpoint, each able to play full
deals through the HTTP API, with unknown or incompatible bots rejected."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bazarblot.core.rules import load_default
from bazarblot.ui import opponents
from bazarblot.ui.app import app
from bazarblot.ui.session import STORE

torch = pytest.importorskip("torch")

client = TestClient(app)


@pytest.fixture
def trained_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    from bazarblot.agents.nn.network import apply_action_prior, build_network
    from bazarblot.learn.checkpoint import save_checkpoint

    rules = load_default()
    path = tmp_path / "run" / "final.pt"
    torch.manual_seed(0)
    net = build_network(rules, hidden=(32,))
    apply_action_prior(net, raise_decay=0.5, capot_bias=-4.0, double_bias=-3.0)
    save_checkpoint(path, net, rules, extra={})
    (tmp_path / "old").mkdir()
    torch.save({"obs_version": "v2", "rules_hash": rules.rules_hash}, tmp_path / "old" / "final.pt")
    monkeypatch.setenv("BAZARBLOT_CHECKPOINTS", str(tmp_path / "*" / "final.pt"))
    opponents.reset_cache()
    yield f"trained:{path}"
    opponents.reset_cache()


def setup_function() -> None:
    STORE._games.clear()


def _play_deals(bot: str, n: int) -> None:
    res = client.post("/api/game", json={"mode": "watch", "match_seed": 3, "bot": bot})
    assert res.status_code == 200, res.text
    sid = res.json()["session_id"]
    assert res.json()["bot"] == bot
    finished = 0
    for _ in range(20 * n):  # a deal may be passed out and redealt; each call is bounded
        res = client.post(f"/api/game/{sid}/autoplay")
        assert res.status_code == 200, res.text
        if res.json()["state"]["phase"] == "TERMINAL":
            finished += 1
            if finished == n:
                return
            client.post(f"/api/game/{sid}/next_deal")
    raise AssertionError(f"only {finished} of {n} deals finished")


def test_listing_keeps_only_loadable_checkpoints(trained_checkpoint: str) -> None:
    ids = [b["id"] for b in client.get("/api/bots").json()["bots"]]
    assert ids == [trained_checkpoint, "heuristic", "simple"]


def test_trained_bot_is_the_default(trained_checkpoint: str) -> None:
    res = client.post("/api/game", json={"mode": "watch", "match_seed": 1})
    assert res.json()["bot"] == trained_checkpoint


@pytest.mark.parametrize("bot", ["heuristic", "simple"])
def test_fixed_bots_play_full_deals(bot: str) -> None:
    _play_deals(bot, 3)


def test_trained_bot_plays_full_deals(trained_checkpoint: str) -> None:
    _play_deals(trained_checkpoint, 3)


def test_trained_bots_partner_a_human(trained_checkpoint: str) -> None:
    res = client.post(
        "/api/game",
        json={"mode": "play", "human_seat": 0, "match_seed": 4, "bot": trained_checkpoint},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["state"]["to_act"] == 0 or data["state"]["phase"] != "AUCTION"


def test_unknown_bot_is_rejected(trained_checkpoint: str) -> None:
    res = client.post("/api/game", json={"mode": "watch", "bot": "trained:nope.pt"})
    assert res.status_code == 400
    res = client.post("/api/game", json={"mode": "watch", "bot": "wizard"})
    assert res.status_code == 400
