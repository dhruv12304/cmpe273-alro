"""ERP Stub — synthetic order generation and A/B comparison (port 8003)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from order_gen import generate_order

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

GRID_SIZE = int(os.getenv("GRID_SIZE", "5"))
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8001")

# ---------------------------------------------------------------------------
# Logging — same JSON formatter as agent-service
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)


_configure_logging()
log = logging.getLogger("erp-stub")

# ---------------------------------------------------------------------------
# In-memory history (newest first, max 200 entries)
# ---------------------------------------------------------------------------

_history: list[dict] = []


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ERP Stub started grid_size=%s agent_url=%s", GRID_SIZE, AGENT_SERVICE_URL)
    yield


app = FastAPI(title="ALRO ERP Stub", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    count: int = 10
    policy: str = "rl"  # 'rl' | 'greedy'


class BatchRequest(BaseModel):
    count: int = 20


class BatchMetrics(BaseModel):
    policy: str
    avg_cost: float
    on_time_rate: float   # fraction [0.0, 1.0]
    total_cost: float
    orders: list[dict]


class BatchComparisonResponse(BaseModel):
    count: int
    greedy: BatchMetrics
    rl: BatchMetrics
    improvement_cost_pct: float    # (greedy_cost - rl_cost) / greedy_cost * 100
    improvement_ontime_pct: float  # rl_ontime_rate - greedy_ontime_rate in pct points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _submit_order(client: httpx.Client, order: dict, policy: str) -> dict:
    """POST one order to the Agent Service /route endpoint."""
    payload = {
        "order_id": order["order_id"],
        "origin": order["origin"],
        "destination": order["destination"],
        "quantity": order["quantity"],
        "priority": order["priority"],
        "deadline": order["deadline"],
        "policy": policy,
    }
    try:
        r = client.post(f"{AGENT_SERVICE_URL}/route", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError as e:
        log.warning("Agent Service unreachable: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="Agent Service unavailable")
    except httpx.HTTPStatusError as e:
        log.warning("Agent Service returned %s: %s", e.response.status_code, e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Agent Service error: {e.response.status_code}")


def _record(entry: dict) -> None:
    _history.insert(0, entry)
    if len(_history) > 200:
        _history.pop()


def _batch_metrics(policy: str, results: list[dict]) -> BatchMetrics:
    costs = [r["estimated_cost"] for r in results]
    ontime = [r["on_time_probability"] for r in results]
    total = sum(costs)
    count = len(results)
    return BatchMetrics(
        policy=policy,
        avg_cost=round(total / count, 2),
        on_time_rate=round(sum(ontime) / count, 4),
        total_cost=round(total, 2),
        orders=results,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "erp-stub"}


@app.post("/orders/generate")
def generate_orders(req: GenerateRequest) -> list[dict]:
    """Generate count orders and submit each to the Agent Service."""
    results = []
    with httpx.Client(timeout=10.0) as client:
        for _ in range(req.count):
            order = generate_order(GRID_SIZE)
            result = _submit_order(client, order, req.policy)
            result["generated_at"] = order["generated_at"]
            _record(result)
            results.append(result)
    return results


@app.post("/orders/batch", response_model=BatchComparisonResponse)
def batch_comparison(req: BatchRequest) -> BatchComparisonResponse:
    """
    Run A/B comparison: same count orders submitted via greedy then via RL.
    Returns side-by-side metrics and improvement percentages.
    """
    orders = [generate_order(GRID_SIZE) for _ in range(req.count)]

    greedy_results: list[dict] = []
    rl_results: list[dict] = []

    with httpx.Client(timeout=10.0) as client:
        for order in orders:
            greedy_order = {**order, "order_id": f"{order['order_id']}-g"}
            result = _submit_order(client, greedy_order, "greedy")
            result["generated_at"] = order["generated_at"]
            _record(result)
            greedy_results.append(result)

        for order in orders:
            rl_order = {**order, "order_id": f"{order['order_id']}-r"}
            result = _submit_order(client, rl_order, "rl")
            result["generated_at"] = order["generated_at"]
            _record(result)
            rl_results.append(result)

    greedy_metrics = _batch_metrics("greedy", greedy_results)
    rl_metrics = _batch_metrics("rl", rl_results)

    greedy_avg = greedy_metrics.avg_cost
    rl_avg = rl_metrics.avg_cost
    improvement_cost_pct = (
        round((greedy_avg - rl_avg) / greedy_avg * 100, 2) if greedy_avg > 0 else 0.0
    )
    improvement_ontime_pct = round(
        (rl_metrics.on_time_rate - greedy_metrics.on_time_rate) * 100, 2
    )

    return BatchComparisonResponse(
        count=req.count,
        greedy=greedy_metrics,
        rl=rl_metrics,
        improvement_cost_pct=improvement_cost_pct,
        improvement_ontime_pct=improvement_ontime_pct,
    )


@app.get("/orders/history")
def get_history() -> list[dict]:
    """Return submitted orders newest first, max 200."""
    return _history[:200]
