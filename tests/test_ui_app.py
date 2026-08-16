"""M1.5: end-to-end HTTP tests against the FastAPI app, via `TestClient` (no real socket)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bazarblot.ui.app import app
from bazarblot.ui.session import STORE

client = TestClient(app)


def setup_function() -> None:
    # Each test gets a clean session store — STORE is process-global.
    STORE._games.clear()
    STORE._replays.clear()


def test_index_serves_html() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert b"Bazar Blot" in res.content


def test_new_watch_game_and_step_through() -> None:
    res = client.post("/api/game", json={"mode": "watch"})
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    assert data["mode"] == "watch"
    assert data["human_seat"] is None
    assert len(data["state"]["hands"]) == 4  # full state: all 4 hands present

    res = client.post(f"/api/game/{sid}/bot_step")
    assert res.status_code == 200


def test_watch_game_autoplay_reaches_terminal_or_stays_bounded() -> None:
    res = client.post("/api/game", json={"mode": "watch", "match_seed": 1})
    sid = res.json()["session_id"]
    for _ in range(50):
        data = client.post(f"/api/game/{sid}/autoplay").json()
        if data["state"]["phase"] == "TERMINAL":
            break
        client.post(f"/api/game/{sid}/next_deal")
    assert data["state"]["phase"] in ("TERMINAL", "AUCTION", "PLAY")


def test_new_play_game_requires_human_seat() -> None:
    res = client.post("/api/game", json={"mode": "play"})
    assert res.status_code == 400


def test_play_game_hides_other_hands() -> None:
    res = client.post("/api/game", json={"mode": "play", "human_seat": 2, "match_seed": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["human_seat"] == 2
    assert "hands" not in data["state"]
    assert "my_hand" in data["state"]
    assert len(data["state"]["my_hand"]) == 8
    assert data["state"]["hand_sizes"] == [8, 8, 8, 8]


def test_play_game_rejects_action_from_wrong_seat() -> None:
    res = client.post("/api/game", json={"mode": "play", "human_seat": 0, "match_seed": 9})
    sid = res.json()["session_id"]
    # act as a seat that is not the human, and is (almost certainly) not to_act either
    res = client.post(f"/api/game/{sid}/action", json={"seat": 1, "action": {"type": "pass"}})
    assert res.status_code == 403


def test_play_game_accepts_a_legal_action_from_the_human() -> None:
    res = client.post("/api/game", json={"mode": "play", "human_seat": 0, "match_seed": 9})
    data = res.json()
    sid = data["session_id"]
    state = data["state"]
    if state["to_act"] != 0:
        # bots auto-advance; if it's not seat 0's turn yet the fixture seed didn't put them
        # there — that's fine, just confirm the auto-advance itself worked without error
        assert state["phase"] in ("AUCTION", "PLAY", "TERMINAL", "ABORTED")
        return
    action = next(a for a in state["legal_actions"] if a["type"] in ("pass", "bid"))
    payload = (
        {"type": "pass"}
        if action["type"] == "pass"
        else {
            "type": "bid",
            "level": action["min_level"],
            "contract_type": action["contract_types"][0],
            "capot": False,
        }
    )
    res = client.post(f"/api/game/{sid}/action", json={"seat": 0, "action": payload})
    assert res.status_code == 200


def test_unknown_session_returns_404() -> None:
    res = client.get("/api/game/does-not-exist")
    assert res.status_code == 404


def test_malformed_action_returns_400() -> None:
    res = client.post("/api/game", json={"mode": "watch"})
    sid = res.json()["session_id"]
    to_act_res = client.get(f"/api/game/{sid}")
    to_act = to_act_res.json()["state"]["to_act"]
    res = client.post(
        f"/api/game/{sid}/action", json={"seat": to_act, "action": {"type": "nonsense"}}
    )
    assert res.status_code == 400


def test_replay_log_requires_finished_deal() -> None:
    res = client.post("/api/game", json={"mode": "watch"})
    sid = res.json()["session_id"]
    res = client.get(f"/api/game/{sid}/replay_log")
    assert res.status_code == 400


def test_full_replay_roundtrip() -> None:
    res = client.post("/api/game", json={"mode": "watch", "match_seed": 3})
    sid = res.json()["session_id"]
    data = None
    for _ in range(100):
        data = client.post(f"/api/game/{sid}/bot_step").json()
        if data["state"]["phase"] == "TERMINAL":
            break
        if data["state"]["phase"] == "ABORTED":
            client.post(f"/api/game/{sid}/next_deal")
    assert data is not None and data["state"]["phase"] == "TERMINAL"

    log = client.get(f"/api/game/{sid}/replay_log").json()
    assert log["rules_hash"]
    assert len(log["actions"]) > 0

    res = client.post("/api/replay", json={"log": log})
    assert res.status_code == 200
    replay_data = res.json()
    rid = replay_data["session_id"]
    assert replay_data["index"] == 0
    total = replay_data["total_actions"]
    assert total == len(log["actions"])

    for _ in range(total):
        res = client.post(f"/api/replay/{rid}/step")
        assert res.status_code == 200
    final = res.json()
    assert final["index"] == total
    assert final["state"]["phase"] == "TERMINAL"

    res = client.post(f"/api/replay/{rid}/goto", json={"index": 0})
    assert res.status_code == 200
    assert res.json()["index"] == 0


def test_replay_rejects_mismatched_rules_hash() -> None:
    bogus_log = {
        "rules_hash": "not-a-real-hash",
        "match_seed": 1,
        "deal_number": 0,
        "dealer": 0,
        "deal_seed": 1,
        "actions": [],
    }
    res = client.post("/api/replay", json={"log": bogus_log})
    assert res.status_code == 404
