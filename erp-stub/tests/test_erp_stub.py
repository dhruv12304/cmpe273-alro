"""Acceptance tests for the ERP Stub (Document 2, Section 5.4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main as erp_main
from main import app
from order_gen import generate_order

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A minimal valid RouteResponse that the Agent Service would return.
_ROUTE_RESPONSE = {
    "order_id": "test-order",
    "route": [0, 1, 2, 3, 4],
    "estimated_cost": 5.20,
    "estimated_steps": 4,
    "on_time_probability": 1.0,
    "explanation": "Route via nodes [0, 1, 2, 3, 4]. Est. cost $5.20.",
    "claude_invoked": False,
    "claude_decision": "n/a",
    "stale_inventory": False,
    "policy": "rl",
}


def _mock_httpx_client(response_body: dict | None = None):
    """Return a context-manager mock for httpx.Client whose .post() returns response_body."""
    body = response_body or _ROUTE_RESPONSE
    mock_response = MagicMock()
    mock_response.json.return_value = body
    mock_response.raise_for_status.return_value = None

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


@pytest.fixture(autouse=True)
def clear_history():
    """Reset in-memory history before every test to prevent cross-test pollution."""
    erp_main._history.clear()
    yield


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# test_generate_valid_orders
# ---------------------------------------------------------------------------


def test_generate_valid_orders(client: TestClient) -> None:
    """POST /orders/generate with count=5 returns array of 5 items each with order_id and route."""
    with patch("main.httpx.Client", return_value=_mock_httpx_client()):
        r = client.post("/orders/generate", json={"count": 5, "policy": "rl"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 5
    for item in data:
        assert "order_id" in item
        assert "route" in item
        assert "estimated_cost" in item


# ---------------------------------------------------------------------------
# test_batch_returns_comparison
# ---------------------------------------------------------------------------


def test_batch_returns_comparison(client: TestClient) -> None:
    """POST /orders/batch with count=10 returns greedy + rl metrics and improvement pcts."""
    greedy_resp = {**_ROUTE_RESPONSE, "policy": "greedy", "estimated_cost": 8.00, "on_time_probability": 0.0}
    rl_resp = {**_ROUTE_RESPONSE, "policy": "rl", "estimated_cost": 5.00, "on_time_probability": 1.0}

    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = MagicMock()
        # First count calls are greedy, next count calls are rl
        mock_response.json.return_value = greedy_resp if call_count <= 10 else rl_resp
        mock_response.raise_for_status.return_value = None
        return mock_response

    mock_client = MagicMock()
    mock_client.post.side_effect = _side_effect
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("main.httpx.Client", return_value=mock_client):
        r = client.post("/orders/batch", json={"count": 10})
    assert r.status_code == 200
    body = r.json()
    assert "greedy" in body
    assert "rl" in body
    assert "improvement_cost_pct" in body
    assert "improvement_ontime_pct" in body
    assert body["count"] == 10
    assert body["greedy"]["policy"] == "greedy"
    assert body["rl"]["policy"] == "rl"
    # Greedy cost > RL cost → positive improvement
    assert body["improvement_cost_pct"] > 0


# ---------------------------------------------------------------------------
# test_high_priority_rate
# ---------------------------------------------------------------------------


def test_high_priority_rate() -> None:
    """In 100 generated orders, between 10% and 30% have priority='high'."""
    orders = [generate_order(grid_size=5) for _ in range(100)]
    high_count = sum(1 for o in orders if o["priority"] == "high")
    assert 10 <= high_count <= 30, (
        f"Expected 10–30 high-priority orders in 100, got {high_count}"
    )


# ---------------------------------------------------------------------------
# test_history_newest_first
# ---------------------------------------------------------------------------


def test_history_newest_first(client: TestClient) -> None:
    """GET /orders/history returns orders in descending generated_at order."""
    import time

    responses = []
    for i in range(3):
        resp = {**_ROUTE_RESPONSE, "order_id": f"order-{i}", "generated_at": f"2026-01-0{i+1}T00:00:00+00:00"}
        responses.append(resp)

    call_idx = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_idx
        mock_response = MagicMock()
        mock_response.json.return_value = {**responses[call_idx], "generated_at": responses[call_idx]["generated_at"]}
        mock_response.raise_for_status.return_value = None
        call_idx += 1
        return mock_response

    mock_client = MagicMock()
    mock_client.post.side_effect = _side_effect
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("main.httpx.Client", return_value=mock_client):
        client.post("/orders/generate", json={"count": 3, "policy": "rl"})

    r = client.get("/orders/history")
    assert r.status_code == 200
    history = r.json()
    assert len(history) == 3
    # History is prepended on each record — newest (last submitted) is first
    assert history[0]["order_id"] == "order-2"
    assert history[2]["order_id"] == "order-0"
