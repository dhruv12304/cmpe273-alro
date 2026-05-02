"""Synthetic order generator for the ALRO ERP Stub."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone


def generate_order(grid_size: int, seed: int | None = None) -> dict:
    """
    Generate one synthetic order for an N×N logistics grid.

    Warehouses occupy the left column (nodes 0, N, 2N, ...).
    Destinations occupy the right column (nodes N-1, 2N-1, ...).
    Priority is 'high' with 20% probability.

    Returns a dict matching the Agent Service RouteRequest schema plus
    a 'generated_at' ISO 8601 UTC timestamp.
    """
    rng = random.Random(seed)

    warehouses = [row * grid_size for row in range(grid_size)]
    destinations = [row * grid_size + (grid_size - 1) for row in range(grid_size)]

    return {
        "order_id": str(uuid.uuid4()),
        "origin": rng.choice(warehouses),
        "destination": rng.choice(destinations),
        "quantity": rng.randint(10, 500),
        "priority": "high" if rng.random() < 0.20 else "normal",
        "deadline": rng.randint(2, 8),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
