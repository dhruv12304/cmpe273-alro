"""LogisticsGrid — N×N grid simulation environment for ALRO."""

from __future__ import annotations

import math
import random
from typing import Optional

from .rewards import PENALTY_PER_STEP, calculate_reward

# Direction order: up, down, left, right (index 0-3)
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class LogisticsGrid:
    """
    N×N grid of logistics nodes. Nodes are indexed 0 to N*N-1.
    Node types: 'warehouse' | 'hub' | 'destination'
    Warehouses: can source orders (have stock) — left column.
    Destinations: order endpoints — right column.
    Hubs: transit-only nodes — interior.
    """

    def __init__(self, n: int = 5, seed: Optional[int] = None):
        if n < 2:
            raise ValueError("Grid size n must be >= 2")
        self._n = n
        self._n_nodes = n * n
        self._seed = seed
        self._rng = random.Random(seed)

        # Node type assignment: left col=warehouse, right col=destination, rest=hub
        self._node_types: dict[int, str] = {}
        for idx in range(n * n):
            _, c = divmod(idx, n)
            if c == 0:
                self._node_types[idx] = "warehouse"
            elif c == n - 1:
                self._node_types[idx] = "destination"
            else:
                self._node_types[idx] = "hub"

        # Node coordinates (row, col)
        self._coords: dict[int, tuple[int, int]] = {
            idx: divmod(idx, n) for idx in range(n * n)
        }

        # Adjacency list and directed edge list
        self._adj: dict[int, list[int]] = {idx: [] for idx in range(n * n)}
        self._edges: list[tuple[int, int]] = []
        for idx in range(n * n):
            r, c = divmod(idx, n)
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    nidx = nr * n + nc
                    self._adj[idx].append(nidx)
                    self._edges.append((idx, nidx))

        # Episode state (populated by reset())
        self._congestion: dict[str, float] = {}
        self._inventory: dict[int, int] = {}
        self._current_node: int = 0
        self._source_node: int = 0
        self._destination_node: int = 0
        self._steps: int = 0
        self._deadline: int = 0
        self._done: bool = True
        self._order_quantity: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        """Total number of possible states = n_nodes * n_nodes."""
        return self._n_nodes * self._n_nodes

    @property
    def n_actions(self) -> int:
        """Number of action slots in the Q-table (= n_nodes for node-indexed actions)."""
        return self._n_nodes

    @property
    def congestion(self) -> dict:
        """Current congestion map. Keys are 'srcIdx-dstIdx' strings."""
        return dict(self._congestion)

    @property
    def inventory(self) -> dict:
        """Current stock levels. Keys are node index ints."""
        return dict(self._inventory)

    def _warehouses(self) -> list[int]:
        return [k for k, v in self._node_types.items() if v == "warehouse"]

    def _destinations(self) -> list[int]:
        return [k for k, v in self._node_types.items() if v == "destination"]

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        origin: Optional[int] = None,
        destination: Optional[int] = None,
        deadline: Optional[int] = None,
    ) -> dict:
        """
        Reset episode state. Randomise congestion weights and stock levels.
        Select random (source_warehouse, destination) pair unless overridden.
        Returns: {state, source, destination, congestion, inventory}
        """
        if seed is not None:
            self._rng = random.Random(seed)

        # Randomise congestion weights
        self._congestion = {
            f"{src}-{dst}": round(self._rng.uniform(0.0, 1.0), 4)
            for src, dst in self._edges
        }

        # Reset stock for all warehouse nodes
        self._inventory = {
            w: self._rng.randint(50, 500) for w in self._warehouses()
        }

        # Select source and destination
        warehouses = self._warehouses()
        destinations = self._destinations()

        self._source_node = origin if origin is not None else self._rng.choice(warehouses)
        self._destination_node = (
            destination if destination is not None else self._rng.choice(destinations)
        )
        self._current_node = self._source_node
        self._steps = 0
        self._deadline = deadline if deadline is not None else self._n * 2
        self._done = False
        self._order_quantity = 0

        return {
            "state": self.encode_state(self._current_node, self._destination_node),
            "source": self._source_node,
            "destination": self._destination_node,
            "congestion": dict(self._congestion),
            "inventory": dict(self._inventory),
        }

    def set_congestion(self, congestion: dict) -> None:
        """Inject congestion weights from the Inventory Service."""
        self._congestion = {str(k): float(v) for k, v in congestion.items()}

    def set_inventory(self, inventory: dict) -> None:
        """Inject inventory levels from the Inventory Service."""
        self._inventory = {int(k): int(v) for k, v in inventory.items()}

    def set_order_quantity(self, quantity: int) -> None:
        """Set order quantity for invalid-warehouse detection during this episode."""
        self._order_quantity = quantity

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """
        Take action (move to adjacent node index).
        Returns: (next_state, reward, done, info)
        Raises ValueError if action is not a valid adjacent node.
        """
        if self._done:
            raise RuntimeError("Episode done — call reset() first")

        valid = self.valid_actions(
            self.encode_state(self._current_node, self._destination_node)
        )
        if action not in valid:
            raise ValueError(
                f"Action {action} is not adjacent to current node {self._current_node}"
            )

        # Check warehouse validity on first step from source
        invalid_warehouse = False
        if (
            self._steps == 0
            and self._node_types.get(self._source_node) == "warehouse"
            and self._order_quantity > 0
        ):
            stock = self._inventory.get(self._source_node, 0)
            if stock < self._order_quantity:
                invalid_warehouse = True

        # Transition cost: euclidean_distance × (1 + congestion)
        edge_key = f"{self._current_node}-{action}"
        cw = self._congestion.get(edge_key, 0.5)
        r1, c1 = self._coords[self._current_node]
        r2, c2 = self._coords[action]
        base_dist = math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
        cost = round(base_dist * (1.0 + cw), 4)

        self._current_node = action
        self._steps += 1

        reached = self._current_node == self._destination_node
        max_exceeded = self._steps >= self._n_nodes
        done = reached or max_exceeded
        on_time = reached and self._steps <= self._deadline

        if done:
            reward = calculate_reward(
                on_time=on_time,
                steps_taken=self._steps,
                deadline=self._deadline,
                invalid_warehouse=invalid_warehouse,
            )
        else:
            reward = PENALTY_PER_STEP

        self._done = done

        return (
            self.encode_state(self._current_node, self._destination_node),
            reward,
            done,
            {"cost": cost, "on_time": on_time, "invalid_warehouse": invalid_warehouse},
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def valid_actions(self, state: int) -> list[int]:
        """Return list of valid next-hop node indices from current state."""
        current_node, _ = self.decode_state(state)
        return list(self._adj[current_node])

    def encode_state(self, current_node: int, destination: int) -> int:
        """Encode (current_node, destination) → flat int for Q-table index."""
        return current_node * self._n_nodes + destination

    def decode_state(self, state: int) -> tuple[int, int]:
        """Decode flat int → (current_node, destination)."""
        return divmod(state, self._n_nodes)

    def node_coords(self, node: int) -> tuple[int, int]:
        """Return (row, col) of node."""
        return self._coords[node]

    def node_type(self, node: int) -> str:
        return self._node_types[node]
