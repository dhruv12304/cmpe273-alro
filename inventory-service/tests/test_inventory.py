"""Acceptance tests for the inventory service (Document 2, Section 4.3)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_get_all_inventory(client: TestClient) -> None:
    """GET /inventory returns dict with N keys (one per warehouse node)."""
    n = 5
    r = client.get("/inventory")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == n
    for key in data:
        assert int(key) in {row * n for row in range(n)}


def test_deduct_stock(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """PUT /inventory/{id} with deduct=100 reduces stock by exactly 100."""
    monkeypatch.setattr("state.random.randint", lambda a, b: 300)
    with TestClient(app) as c:
        r = c.put("/inventory/0", json={"deduct": 100})
        assert r.status_code == 200
        body = r.json()
        assert body["node_id"] == 0
        assert body["new_stock"] == 200


def test_stock_floor_zero(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deducting more than available stock results in 0, not negative."""
    monkeypatch.setattr("state.random.randint", lambda a, b: 50)
    with TestClient(app) as c:
        r = c.put("/inventory/0", json={"deduct": 1000})
        assert r.status_code == 200
        assert r.json()["new_stock"] == 0


def test_congestion_reset(client: TestClient) -> None:
    """POST /congestion/reset returns new values; at least one value differs from previous."""
    before = client.get("/congestion").json()
    r = client.post("/congestion/reset")
    assert r.status_code == 200
    after = r.json()
    assert set(before.keys()) == set(after.keys())
    assert any(
        round(float(before[k]), 2) != round(float(after[k]), 2) for k in before
    )


def test_health_check(client: TestClient) -> None:
    """GET /health returns 200 with status='ok'."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "inventory"


def test_put_deduct_negative_returns_422(client: TestClient) -> None:
    r = client.put("/inventory/0", json={"deduct": -1})
    assert r.status_code == 422


def test_get_non_warehouse_returns_404(client: TestClient) -> None:
    r = client.get("/inventory/1")
    assert r.status_code == 404
