"""Claude AI reasoning integration for ALRO Agent Service."""

from __future__ import annotations

import re
import time

import anthropic

SYSTEM_PROMPT = """You are a logistics routing advisor for an automated warehouse system.
Evaluate the proposed delivery route and respond in this exact format:
DECISION: [approve/reroute/escalate]
REASON: [1-2 sentences, max 60 words]
Do not add any other text. Use 'approve' if the route is acceptable,
'reroute' if a better option exists, 'escalate' if human review is needed."""

USER_PROMPT_TEMPLATE = """
ORDER:
  ID: {order_id}
  Origin warehouse: Node {origin}
  Destination: Node {destination}
  Quantity: {quantity} units
  Priority: {priority}
  Deadline: {deadline} steps

TOP ROUTE OPTIONS (Q-learning ranked):
{route_options}

CURRENT CONDITIONS:
  Route congestion weights: {congestion_summary}
  Origin warehouse stock: {stock_level} units (order needs {quantity})
"""

_DECISION_RE = re.compile(r"DECISION:\s*(approve|reroute|escalate)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE | re.DOTALL)


class ClaudeAdvisor:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 150,
        timeout_seconds: float = 3.0,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds

    def should_invoke(
        self,
        confidence_spread: float,
        is_high_priority: bool,
        threshold: float,
    ) -> bool:
        """Return True if Claude should be called."""
        return is_high_priority or confidence_spread < threshold

    def advise(
        self,
        order: dict,
        top_routes: list[dict],
        congestion: dict,
        inventory: dict,
    ) -> dict:
        """
        Build structured prompt, call Anthropic API, parse response.
        Returns fallback dict on timeout or API error.
        """
        t0 = time.monotonic()
        try:
            prompt = self._build_prompt(order, top_routes, congestion, inventory)
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._timeout,
            )
            raw = msg.content[0].text if msg.content else ""
            latency_ms = (time.monotonic() - t0) * 1000

            decision_match = _DECISION_RE.search(raw)
            reason_match = _REASON_RE.search(raw)

            decision = decision_match.group(1).lower() if decision_match else "approve"
            reason = reason_match.group(1).strip() if reason_match else raw.strip()
            # Truncate reason to ~80 words
            words = reason.split()
            if len(words) > 80:
                reason = " ".join(words[:80]) + "…"

            return {
                "decision": decision,
                "reason": reason,
                "invoked": True,
                "latency_ms": round(latency_ms, 1),
            }

        except Exception:
            return {
                "decision": "approve",
                "reason": "AI reasoning unavailable — proceeding with Q-table decision.",
                "invoked": False,
                "latency_ms": 0.0,
            }

    def _build_prompt(
        self,
        order: dict,
        top_routes: list[dict],
        congestion: dict,
        inventory: dict,
    ) -> str:
        """Build the structured prompt string per spec template."""
        # Format route options
        route_lines = []
        for i, r in enumerate(top_routes, start=1):
            node_id = r.get("node_id", r.get("action", "?"))
            q_val = r.get("q_value", 0.0)
            est_cost = abs(q_val) if q_val != 0 else 1.0
            route_lines.append(
                f"{i}. Node {order.get('origin')} → {node_id}  "
                f"Q-value: {q_val:.3f}  Est. cost: ${est_cost:.2f}"
            )
        route_options = "\n".join(route_lines) if route_lines else "N/A"

        # Congestion summary (top 5 relevant edges)
        cong_items = sorted(congestion.items(), key=lambda x: x[1], reverse=True)[:5]
        congestion_summary = ", ".join(f"{k}={v:.2f}" for k, v in cong_items) or "N/A"

        origin = order.get("origin", 0)
        stock_level = inventory.get(origin, inventory.get(str(origin), 0))

        return USER_PROMPT_TEMPLATE.format(
            order_id=order.get("order_id", "N/A"),
            origin=origin,
            destination=order.get("destination", 0),
            quantity=order.get("quantity", 0),
            priority=order.get("priority", "normal"),
            deadline=order.get("deadline", 0),
            route_options=route_options,
            congestion_summary=congestion_summary,
            stock_level=stock_level,
        )
