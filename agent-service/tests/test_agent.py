"""Acceptance tests for the Agent Service (Document 2, Section 3.6)."""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Make environment/ importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import main as agent_main
from environment.grid import LogisticsGrid
from main import app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_INVENTORY_PAYLOAD = {
    "inventory": {"0": 300, "5": 200, "10": 400, "15": 150, "20": 250},
    "congestion": {},
}

_BASE_ORDER = {
    "order_id": "test-1",
    "origin": 0,
    "destination": 4,
    "quantity": 50,
    "priority": "normal",
    "deadline": 8,
}


@pytest.fixture
def client():
    """TestClient with lifespan — initialises grid, agent, advisor, cache."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_agent_state(client):
    """Reset Q-table and stats before every test to guarantee isolation."""
    client.post("/reset")
    yield


@pytest.fixture
def mock_inventory():
    """Patch _fetch_inventory so tests don't need the Inventory Service running."""
    with patch("main._fetch_inventory", new=AsyncMock(return_value=_INVENTORY_PAYLOAD)):
        yield


# ---------------------------------------------------------------------------
# test_route_returns_valid_path
# ---------------------------------------------------------------------------


def test_route_returns_valid_path(client, mock_inventory) -> None:
    """POST /route returns a route where each consecutive node is adjacent in the grid."""
    r = client.post("/route", json=_BASE_ORDER)
    assert r.status_code == 200
    body = r.json()
    route = body["route"]
    assert len(route) >= 2

    grid = LogisticsGrid(n=5)
    for i in range(len(route) - 1):
        state = grid.encode_state(route[i], _BASE_ORDER["destination"])
        assert route[i + 1] in grid.valid_actions(state), (
            f"Node {route[i+1]} is not adjacent to {route[i]}"
        )


# ---------------------------------------------------------------------------
# test_train_reward_improves
# ---------------------------------------------------------------------------


def test_train_reward_improves(client) -> None:
    """After 500 training episodes, avg reward of last 50 > avg reward of first 50."""
    r = client.post("/train", json={"episodes": 500})
    assert r.status_code == 200
    curve = r.json()["reward_curve"]
    assert len(curve) == 500

    avg_first_50 = sum(curve[:50]) / 50
    avg_last_50 = sum(curve[-50:]) / 50
    assert avg_last_50 > avg_first_50, (
        f"Reward did not improve: first-50 avg={avg_first_50:.2f}, last-50 avg={avg_last_50:.2f}"
    )


# ---------------------------------------------------------------------------
# test_stale_cache_on_inventory_down
# ---------------------------------------------------------------------------


def test_stale_cache_on_inventory_down(client) -> None:
    """With Inventory Service failing, /route returns 200 with stale_inventory=true."""
    # Pre-populate cache with valid data
    agent_main.cache.set(_INVENTORY_PAYLOAD)
    # Backdate timestamp past the 60s TTL so cache is considered stale
    agent_main.cache._timestamp -= 120

    with patch("main._fetch_inventory", new=AsyncMock(side_effect=Exception("inventory down"))):
        r = client.post("/route", json=_BASE_ORDER)

    assert r.status_code == 200
    assert r.json()["stale_inventory"] is True


# ---------------------------------------------------------------------------
# test_claude_invoked_high_priority
# ---------------------------------------------------------------------------


def test_claude_invoked_high_priority(client, mock_inventory) -> None:
    """An order with priority='high' results in claude_invoked=true in the response."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="DECISION: approve\nREASON: Route looks optimal.")]

    with patch.object(agent_main.advisor._client.messages, "create", return_value=mock_msg):
        r = client.post("/route", json={**_BASE_ORDER, "priority": "high"})

    assert r.status_code == 200
    assert r.json()["claude_invoked"] is True


# ---------------------------------------------------------------------------
# test_claude_fallback_on_timeout
# ---------------------------------------------------------------------------


def test_claude_fallback_on_timeout(client, mock_inventory) -> None:
    """With Claude API timing out, /route returns 200 with claude_invoked=false."""
    with patch.object(
        agent_main.advisor._client.messages,
        "create",
        side_effect=Exception("timeout"),
    ):
        r = client.post("/route", json={**_BASE_ORDER, "priority": "high"})

    assert r.status_code == 200
    body = r.json()
    assert body["claude_invoked"] is False
    # main.py only uses claude_result["reason"] when invoked=True;
    # on fallback it uses the Q-table route string instead.
    assert "Route via nodes" in body["explanation"]


# ---------------------------------------------------------------------------
# test_reset_clears_qtable
# ---------------------------------------------------------------------------


def test_reset_clears_qtable(client) -> None:
    """POST /reset followed by GET /stats returns episode_count=0 and epsilon=EPSILON_START."""
    # Train briefly so there's state to reset
    client.post("/train", json={"episodes": 10})

    client.post("/reset")

    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["episode_count"] == 0
    assert body["epsilon"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_rl_beats_greedy_bulk  (integration — requires full stack)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rl_beats_greedy_bulk() -> None:
    """
    RL agent average cost < greedy average cost over 50 orders via /orders/batch.

    Requires:
      - Agent Service trained (run `make train` first)
      - ERP Stub running at ERP_STUB_URL (default http://localhost:8003)
    This test is skipped in unit-test runs: pytest -m "not integration"
    """
    import httpx as _httpx

    erp_url = os.getenv("ERP_STUB_URL", "http://localhost:8003")
    r = _httpx.post(f"{erp_url}/orders/batch", json={"count": 50}, timeout=120.0)
    assert r.status_code == 200
    body = r.json()
    assert body["rl"]["avg_cost"] < body["greedy"]["avg_cost"], (
        f"RL avg_cost {body['rl']['avg_cost']:.2f} did not beat greedy {body['greedy']['avg_cost']:.2f}. "
        "Ensure the agent is trained before running this test."
    )
