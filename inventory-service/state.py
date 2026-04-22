"""In-memory warehouse state for the inventory service."""

from __future__ import annotations

import random
from typing import Dict, Set


def _warehouse_node_indices(n: int) -> Set[int]:
    """Left column of an N×N grid: 0, N, 2N, …, (N-1)*N."""
    return {row * n for row in range(n)}


def _directed_grid_edges(n: int) -> list[tuple[int, int]]:
    """4-neighbor directed edges on an N×N grid (both directions as separate edges)."""
    edges: list[tuple[int, int]] = []
    for idx in range(n * n):
        r, c = divmod(idx, n)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                nidx = nr * n + nc
                edges.append((idx, nidx))
    return edges


class WarehouseState:
    """
    N×N grid warehouse state.

    Warehouses occupy the left column: node indices 0, N, 2N, …, (N-1)*N.
    """

    def __init__(self, grid_size: int):
        if grid_size < 1:
            raise ValueError("grid_size must be >= 1")
        self._n = grid_size
        self._warehouses = _warehouse_node_indices(grid_size)
        self._stock: Dict[int, int] = {}
        self._congestion: Dict[str, float] = {}
        self._edge_pairs = _directed_grid_edges(grid_size)
        self.reset_stock()
        self._init_congestion()

    def _init_congestion(self) -> None:
        self._congestion = {
            f"{src}-{dst}": random.uniform(0.0, 1.0) for src, dst in self._edge_pairs
        }

    def get_all_inventory(self) -> dict:
        """Return { node_id: stock_level } for all warehouse nodes."""
        return dict(self._stock)

    def get_node(self, node_id: int) -> dict:
        """
        Return { 'node_id': int, 'stock': int, 'type': str }.
        Raises KeyError if node_id not a warehouse.
        """
        if node_id not in self._warehouses:
            raise KeyError(node_id)
        return {"node_id": node_id, "stock": self._stock[node_id], "type": "warehouse"}

    def deduct_stock(self, node_id: int, quantity: int) -> int:
        """
        Deduct quantity from node stock. Floor at 0.
        Returns new stock level.
        Raises KeyError if not a warehouse node.
        """
        if node_id not in self._warehouses:
            raise KeyError(node_id)
        new_level = max(0, self._stock[node_id] - quantity)
        self._stock[node_id] = new_level
        return new_level

    def get_congestion(self) -> dict:
        """Return { 'srcIdx-dstIdx': float } for all edges."""
        return dict(self._congestion)

    def reset_congestion(self) -> dict:
        """Randomise all edge weights. Returns new congestion map."""
        self._init_congestion()
        return self.get_congestion()

    def reset_stock(self) -> None:
        """Re-initialise all stock to fresh random values."""
        self._stock = {
            w: random.randint(50, 500) for w in sorted(self._warehouses)
        }
