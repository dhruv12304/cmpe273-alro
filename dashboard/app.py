"""ALRO Dashboard — live metrics via 2-second HTTP polling."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8001")
INV_URL = os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8002")
ERP_URL = os.getenv("ERP_STUB_URL", "http://localhost:8003")
POLL_MS = 2000

st.set_page_config(page_title="ALRO Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_safe(url: str, timeout: int = 2) -> dict | None:
    """GET url. Return parsed JSON or None on any error."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def post_safe(url: str, payload: dict | None = None, timeout: int = 60) -> dict | None:
    """POST url. Return parsed JSON or None on any error."""
    try:
        r = requests.post(url, json=payload or {}, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _health_indicator(label: str, healthy: bool) -> None:
    colour = "🟢" if healthy else "🔴"
    st.metric(label=label, value=f"{colour} {'OK' if healthy else 'DOWN'}")


def _draw_congestion_map(congestion: dict, grid_size: int) -> plt.Figure:
    """Render average outgoing congestion per node as a grid heatmap."""
    n = grid_size
    totals = np.zeros((n, n))
    counts = np.zeros((n, n))

    for edge_key, weight in congestion.items():
        try:
            src, _ = map(int, edge_key.split("-"))
            r, c = divmod(src, n)
            totals[r, c] += float(weight)
            counts[r, c] += 1
        except (ValueError, TypeError):
            continue

    matrix = np.where(counts > 0, totals / counts, 0.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Avg congestion")
    ax.set_title("Congestion map (avg per node)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    node_labels = {0: "W", n - 1: "D"}
    for row in range(n):
        for col in range(n):
            label = node_labels.get(col, "H")
            ax.text(col, row, label, ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Auto-refresh polling
# ---------------------------------------------------------------------------

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "last_batch" not in st.session_state:
    st.session_state.last_batch = None
if "last_train_curve" not in st.session_state:
    st.session_state.last_train_curve = []

if st.session_state.auto_refresh:
    st.session_state.stats = fetch_safe(f"{AGENT_URL}/stats")
    st.session_state.health_a = fetch_safe(f"{AGENT_URL}/health")
    st.session_state.health_i = fetch_safe(f"{INV_URL}/health")
    st.session_state.health_e = fetch_safe(f"{ERP_URL}/health")
    st.session_state.cong = fetch_safe(f"{INV_URL}/congestion")
    time.sleep(POLL_MS / 1000)
    st.rerun()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ALRO Controls")
    st.session_state.auto_refresh = st.checkbox(
        "Auto-refresh (2s)", value=st.session_state.auto_refresh
    )

    st.divider()

    if st.button("▶ Train 200 Episodes"):
        with st.spinner("Training…"):
            result = post_safe(f"{AGENT_URL}/train", {"episodes": 200}, timeout=120)
        if result:
            st.session_state.last_train_curve = result.get("reward_curve", [])
            st.session_state.stats = fetch_safe(f"{AGENT_URL}/stats")
            st.success(f"Done — ε={result.get('final_epsilon', '?'):.3f}")
        else:
            st.error("Train call failed")

    if st.button("📦 Run A/B Batch (20 orders)"):
        with st.spinner("Running batch…"):
            result = post_safe(f"{ERP_URL}/orders/batch", {"count": 20}, timeout=120)
        if result:
            st.session_state.last_batch = result
            st.success(
                f"RL cost improvement: {result.get('improvement_cost_pct', 0):.1f}%"
            )
        else:
            st.error("Batch call failed")

    if st.button("🔄 Reset Agent"):
        post_safe(f"{AGENT_URL}/reset")
        st.session_state.last_train_curve = []
        st.session_state.stats = fetch_safe(f"{AGENT_URL}/stats")
        st.info("Agent reset")

    if st.button("🌐 Reset Congestion"):
        post_safe(f"{INV_URL}/congestion/reset")
        st.session_state.cong = fetch_safe(f"{INV_URL}/congestion")
        st.info("Congestion randomised")

# ---------------------------------------------------------------------------
# Fetch latest data if not already populated this run
# ---------------------------------------------------------------------------

if "stats" not in st.session_state:
    st.session_state.stats = fetch_safe(f"{AGENT_URL}/stats")
if "cong" not in st.session_state:
    st.session_state.cong = fetch_safe(f"{INV_URL}/congestion")
if "health_a" not in st.session_state:
    st.session_state.health_a = fetch_safe(f"{AGENT_URL}/health")
if "health_i" not in st.session_state:
    st.session_state.health_i = fetch_safe(f"{INV_URL}/health")
if "health_e" not in st.session_state:
    st.session_state.health_e = fetch_safe(f"{ERP_URL}/health")

stats = st.session_state.stats or {}
cong = st.session_state.cong or {}

# ---------------------------------------------------------------------------
# Row 1: Reward Curve | A/B Comparison
# ---------------------------------------------------------------------------

st.header("ALRO — Autonomous Logistics & Routing Optimizer")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reward Curve")
    curve = st.session_state.last_train_curve
    if curve:
        # Smooth with a rolling average per 10 episodes
        window = 10
        smoothed = [
            sum(curve[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(curve))
        ]
        st.line_chart({"avg reward": smoothed})
        st.caption(
            f"Episodes: {len(curve)} | "
            f"Last-50 avg: {sum(curve[-50:]) / min(len(curve), 50):.1f}"
        )
    else:
        st.info("Run 'Train 200 Episodes' to populate the reward curve.")

with col2:
    st.subheader("A/B Comparison — Greedy vs RL")
    batch = st.session_state.last_batch
    if batch:
        greedy = batch.get("greedy", {})
        rl = batch.get("rl", {})
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        axes[0].bar(["Greedy", "RL"], [greedy.get("avg_cost", 0), rl.get("avg_cost", 0)],
                    color=["#e06c75", "#98c379"])
        axes[0].set_title("Avg Delivery Cost")
        axes[0].set_ylabel("Cost")

        axes[1].bar(["Greedy", "RL"],
                    [greedy.get("on_time_rate", 0) * 100, rl.get("on_time_rate", 0) * 100],
                    color=["#e06c75", "#98c379"])
        axes[1].set_title("On-time Rate (%)")
        axes[1].set_ylim(0, 100)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Cost improvement: **{batch.get('improvement_cost_pct', 0):.1f}%** | "
            f"On-time delta: **{batch.get('improvement_ontime_pct', 0):+.1f} pp**"
        )
    else:
        st.info("Run 'A/B Batch' to populate the comparison chart.")

# ---------------------------------------------------------------------------
# Row 2: Recent Decisions
# ---------------------------------------------------------------------------

st.subheader("Recent Decisions")
decisions = stats.get("recent_decisions", [])
if decisions:
    rows = []
    for d in decisions:
        rows.append({
            "Order ID": str(d.get("order_id", ""))[:18],
            "Route": str(d.get("route", "")),
            "Cost": f"${d.get('estimated_cost', 0):.2f}",
            "On-time": "✓" if d.get("on_time_probability", 0) >= 1.0 else "✗",
            "Claude?": "Yes" if d.get("claude_invoked") else "No",
            "Explanation": str(d.get("explanation", ""))[:80],
        })
    st.dataframe(rows, use_container_width=True)
else:
    st.info("No routing decisions yet — submit an order via the Agent Service.")

# ---------------------------------------------------------------------------
# Row 3: Service Health
# ---------------------------------------------------------------------------

st.subheader("Service Health")
h_col1, h_col2, h_col3, h_col4 = st.columns(4)

with h_col1:
    _health_indicator("Agent Service :8001", st.session_state.health_a is not None)
with h_col2:
    inv_data = st.session_state.get("health_i")
    inv_degraded = (
        stats.get("inventory_service_healthy") is False and
        stats.get("stale_cache_hits", 0) > 0
    )
    label = "Inventory :8002" + (" (degraded)" if inv_degraded else "")
    _health_indicator(label, inv_data is not None)
with h_col3:
    _health_indicator("ERP Stub :8003", st.session_state.health_e is not None)
with h_col4:
    episode_count = stats.get("episode_count", 0)
    st.metric("Episodes trained", episode_count)

# ---------------------------------------------------------------------------
# Row 4: Congestion Map
# ---------------------------------------------------------------------------

st.subheader("Congestion Map")
if cong:
    grid_size = int(os.getenv("GRID_SIZE", "5"))
    fig = _draw_congestion_map(cong, grid_size)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("W = Warehouse (left col) | H = Hub (interior) | D = Destination (right col)")
else:
    st.info("Inventory Service unreachable — congestion map unavailable.")
