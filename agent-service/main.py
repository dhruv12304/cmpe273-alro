"""Agent Service — Q-learning + Claude AI reasoning (port 8001)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Make the sibling environment/ package importable when running locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.grid import LogisticsGrid
from environment.baseline import greedy_route
from cache import InventoryCache
from claude_advisor import ClaudeAdvisor
from q_agent import QAgent

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

GRID_SIZE = int(os.getenv("GRID_SIZE", 5))
ALPHA = float(os.getenv("ALPHA", 0.1))
GAMMA = float(os.getenv("GAMMA", 0.95))
EPSILON_START = float(os.getenv("EPSILON_START", 1.0))
EPSILON_DECAY = float(os.getenv("EPSILON_DECAY", 0.995))
EPSILON_MIN = float(os.getenv("EPSILON_MIN", 0.05))
TRAINING_EPISODES = int(os.getenv("TRAINING_EPISODES", 500))
INVENTORY_SERVICE_URL = os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8002")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 60))
CLAUDE_CONFIDENCE_THRESHOLD = float(os.getenv("CLAUDE_CONFIDENCE_THRESHOLD", 0.10))
CLAUDE_TIMEOUT_SECONDS = float(os.getenv("CLAUDE_TIMEOUT_SECONDS", 3.0))
QTABLE_PATH = os.getenv("QTABLE_PATH", "/data/qtable.npy")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Logging
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
log = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

grid: LogisticsGrid
agent: QAgent
advisor: ClaudeAdvisor
cache: InventoryCache

# In-memory stats
_stats: dict[str, Any] = {
    "episode_count": 0,
    "reward_history": [],   # list of per-episode total rewards (for training)
    "total_orders_routed": 0,
    "claude_invocations": 0,
    "stale_cache_hits": 0,
    "recent_decisions": [],
}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global grid, agent, advisor, cache
    grid = LogisticsGrid(n=GRID_SIZE)
    agent = QAgent(
        n_states=grid.n_states,
        n_actions=grid.n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        qtable_path=QTABLE_PATH,
    )
    advisor = ClaudeAdvisor(
        api_key=ANTHROPIC_API_KEY,
        timeout_seconds=CLAUDE_TIMEOUT_SECONDS,
    )
    cache = InventoryCache(ttl_seconds=CACHE_TTL_SECONDS)
    log.info("Agent Service started grid_size=%s", GRID_SIZE)
    yield


app = FastAPI(title="ALRO Agent Service", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    order_id: str
    origin: int
    destination: int
    quantity: int
    priority: str = "normal"   # 'normal' | 'high'
    deadline: int
    policy: str = "rl"         # 'rl' | 'greedy'


class RouteResponse(BaseModel):
    order_id: str
    route: list[int]
    estimated_cost: float
    estimated_steps: int
    on_time_probability: float
    explanation: str
    claude_invoked: bool
    claude_decision: str       # 'approve' | 'reroute' | 'escalate' | 'n/a'
    stale_inventory: bool
    policy: str


class TrainRequest(BaseModel):
    episodes: int = 500
    policy: str = "rl"


class TrainResponse(BaseModel):
    episodes_run: int
    reward_curve: list[float]
    final_epsilon: float
    duration_seconds: float


class StatsResponse(BaseModel):
    episode_count: int
    epsilon: float
    avg_reward_last_50: float
    total_orders_routed: int
    claude_invocations: int
    inventory_service_healthy: bool
    stale_cache_hits: int
    recent_decisions: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fetch_inventory() -> dict:
    """Fetch inventory + congestion from the Inventory Service."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        inv_r = await client.get(f"{INVENTORY_SERVICE_URL}/inventory")
        inv_r.raise_for_status()
        cong_r = await client.get(f"{INVENTORY_SERVICE_URL}/congestion")
        cong_r.raise_for_status()
    return {"inventory": inv_r.json(), "congestion": cong_r.json()}


async def _check_inventory_healthy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{INVENTORY_SERVICE_URL}/health")
            return r.status_code == 200
    except Exception:
        return False


def _record_decision(decision_dict: dict) -> None:
    _stats["recent_decisions"].append(decision_dict)
    if len(_stats["recent_decisions"]) > 10:
        _stats["recent_decisions"].pop(0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    inv_ok = await _check_inventory_healthy()
    return {"status": "ok", "service": "agent", "inventory_reachable": inv_ok}


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest) -> RouteResponse:
    global _stats

    # 1. Fetch inventory with cache fallback
    inventory_data, is_stale = cache.get()
    if inventory_data is None or is_stale:
        try:
            fetched = await _fetch_inventory()
            cache.set(fetched)
            inventory_data = fetched
            is_stale = False
        except Exception:
            if inventory_data is None:
                raise HTTPException(
                    status_code=503,
                    detail="Inventory unavailable and no cache",
                )
            # use stale cache; is_stale remains True

    if is_stale:
        _stats["stale_cache_hits"] += 1

    live_inventory: dict = inventory_data.get("inventory", {})
    live_congestion: dict = inventory_data.get("congestion", {})

    # 2. Route via greedy or RL
    if req.policy == "greedy":
        order_dict = req.model_dump()
        result = greedy_route(grid, order_dict)
        route_nodes = result["route"]
        total_cost = result["cost"]
        on_time = result["on_time"]
        explanation = (
            f"Greedy route via nodes {route_nodes}. "
            f"Est. cost ${total_cost:.2f}."
        )
        resp = RouteResponse(
            order_id=req.order_id,
            route=route_nodes,
            estimated_cost=total_cost,
            estimated_steps=result["steps"],
            on_time_probability=1.0 if on_time else 0.0,
            explanation=explanation,
            claude_invoked=False,
            claude_decision="n/a",
            stale_inventory=is_stale,
            policy="greedy",
        )
        _stats["total_orders_routed"] += 1
        _record_decision(resp.model_dump())
        return resp

    # --- RL policy ---

    # 3. Reset grid for this order and inject live data
    obs = grid.reset(
        origin=req.origin,
        destination=req.destination,
        deadline=req.deadline,
    )
    grid.set_congestion(live_congestion)
    grid.set_inventory(live_inventory)
    grid.set_order_quantity(req.quantity)

    # 4. Run Q-policy (epsilon=0 — pure exploitation)
    route_nodes: list[int] = [req.origin]
    total_cost = 0.0
    on_time = False
    state = grid.encode_state(req.origin, req.destination)
    last_info: dict = {}

    for _ in range(grid.n_states):
        actions = grid.valid_actions(state)
        if not actions:
            break
        action = agent.choose_action(state, actions, training=False)
        next_state, _, done, info = grid.step(action)
        route_nodes.append(action)
        total_cost += info["cost"]
        last_info = info
        state = next_state
        if done:
            on_time = info["on_time"]
            break

    # 5. Claude trigger check (use spread at final state)
    actions_at_state = grid.valid_actions(state)
    spread = (
        agent.confidence_spread(state, actions_at_state)
        if actions_at_state
        else 1.0
    )
    is_high_priority = req.priority == "high"

    claude_result: dict = {"decision": "n/a", "reason": "", "invoked": False}
    if advisor.should_invoke(spread, is_high_priority, CLAUDE_CONFIDENCE_THRESHOLD):
        top_routes = agent.top_actions(
            state, actions_at_state or [req.destination], k=3
        )
        claude_result = advisor.advise(
            order=req.model_dump(),
            top_routes=top_routes,
            congestion=live_congestion,
            inventory=live_inventory,
        )
        if claude_result["invoked"]:
            _stats["claude_invocations"] += 1

    # 6. Build explanation
    if claude_result["invoked"] and claude_result["reason"]:
        explanation = claude_result["reason"]
    else:
        explanation = (
            f"Route via nodes {route_nodes}. "
            f"Est. cost ${total_cost:.2f}."
        )
        if is_stale:
            explanation += " [stale inventory data]"

    resp = RouteResponse(
        order_id=req.order_id,
        route=route_nodes,
        estimated_cost=round(total_cost, 2),
        estimated_steps=len(route_nodes) - 1,
        on_time_probability=1.0 if on_time else 0.0,
        explanation=explanation,
        claude_invoked=claude_result["invoked"],
        claude_decision=claude_result.get("decision", "n/a"),
        stale_inventory=is_stale,
        policy="rl",
    )
    _stats["total_orders_routed"] += 1
    _record_decision(resp.model_dump())
    return resp


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    """Run N training episodes against the local grid environment."""
    t0 = time.monotonic()
    reward_curve: list[float] = []

    for ep in range(req.episodes):
        obs = grid.reset()
        state = obs["state"]
        episode_reward = 0.0

        for _ in range(grid.n_states):
            actions = grid.valid_actions(state)
            if not actions:
                break
            action = agent.choose_action(state, actions, training=True)
            next_state, reward, done, _ = grid.step(action)
            next_actions = grid.valid_actions(next_state)
            agent.update(state, action, reward, next_state, next_actions)
            state = next_state
            episode_reward += reward
            if done:
                break

        agent.decay_epsilon()
        reward_curve.append(episode_reward)

    _stats["episode_count"] += req.episodes
    _stats["reward_history"].extend(reward_curve)

    return TrainResponse(
        episodes_run=req.episodes,
        reward_curve=reward_curve,
        final_epsilon=agent.epsilon,
        duration_seconds=round(time.monotonic() - t0, 2),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    inv_ok = await _check_inventory_healthy()
    history = _stats["reward_history"]
    avg_last_50 = (
        sum(history[-50:]) / min(len(history), 50) if history else 0.0
    )
    return StatsResponse(
        episode_count=_stats["episode_count"],
        epsilon=agent.epsilon,
        avg_reward_last_50=round(avg_last_50, 2),
        total_orders_routed=_stats["total_orders_routed"],
        claude_invocations=_stats["claude_invocations"],
        inventory_service_healthy=inv_ok,
        stale_cache_hits=_stats["stale_cache_hits"],
        recent_decisions=list(_stats["recent_decisions"]),
    )


@app.post("/reset")
def reset_agent():
    """Reset Q-table to zeros (used in live demo to show learning from scratch)."""
    agent.reset()
    _stats["episode_count"] = 0
    _stats["reward_history"] = []
    _stats["total_orders_routed"] = 0
    _stats["claude_invocations"] = 0
    _stats["stale_cache_hits"] = 0
    _stats["recent_decisions"] = []
    return {"status": "reset", "message": "Q-table zeroed and epsilon reset to initial value."}
