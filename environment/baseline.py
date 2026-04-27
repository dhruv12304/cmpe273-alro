"""Greedy baseline policy for ALRO."""

from __future__ import annotations

import math

from .grid import LogisticsGrid


def greedy_route(grid: LogisticsGrid, order: dict) -> dict:
    """
    Deterministic greedy policy.
    At each step, selects the adjacent node with the shortest
    Euclidean distance to the destination.
    Ignores congestion weights and inventory levels.

    Returns: {route, cost, steps, on_time, policy}
    """
    origin: int = order["origin"]
    destination: int = order["destination"]
    deadline: int = order.get("deadline", grid._n * 2)
    quantity: int = order.get("quantity", 0)

    grid.reset(origin=origin, destination=destination, deadline=deadline)
    grid.set_order_quantity(quantity)

    dest_r, dest_c = grid.node_coords(destination)
    route = [origin]
    total_cost = 0.0
    state = grid.encode_state(origin, destination)

    for _ in range(grid.n_states):
        actions = grid.valid_actions(state)
        # Pick neighbor closest to destination by euclidean distance
        best_action = min(
            actions,
            key=lambda a: math.sqrt(
                (grid.node_coords(a)[0] - dest_r) ** 2
                + (grid.node_coords(a)[1] - dest_c) ** 2
            ),
        )
        state, _, done, info = grid.step(best_action)
        route.append(best_action)
        total_cost += info["cost"]
        if done:
            break

    steps = len(route) - 1
    on_time = (route[-1] == destination) and (steps <= deadline)

    return {
        "route": route,
        "cost": round(total_cost, 2),
        "steps": steps,
        "on_time": on_time,
        "policy": "greedy",
    }
