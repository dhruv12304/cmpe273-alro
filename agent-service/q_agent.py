"""Q-learning agent for ALRO."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


class QAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
        qtable_path: str,
    ):
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon_init = epsilon
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._qtable_path = qtable_path
        self._episode_count = 0

        if os.path.exists(qtable_path):
            loaded = np.load(qtable_path)
            if loaded.shape == (n_states, n_actions):
                self._q = loaded
            else:
                self._q = np.zeros((n_states, n_actions), dtype=np.float64)
        else:
            self._q = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def episode_count(self) -> int:
        return self._episode_count

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(
        self,
        state: int,
        valid_actions: list[int],
        training: bool = True,
    ) -> int:
        """
        Epsilon-greedy selection over valid_actions only.
        Masked: invalid actions are never selected regardless of Q-value.
        When training=False, always picks the greedy best action (epsilon=0).
        """
        if training and random.random() < self._epsilon:
            return random.choice(valid_actions)
        return self._greedy_action(state, valid_actions)

    def _greedy_action(self, state: int, valid_actions: list[int]) -> int:
        q_vals = self._q[state]
        return max(valid_actions, key=lambda a: q_vals[a])

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_valid_actions: list[int],
    ) -> None:
        """Bellman update. Saves Q-table to disk after every update."""
        if next_valid_actions:
            best_next_q = max(self._q[next_state, a] for a in next_valid_actions)
        else:
            best_next_q = 0.0

        td_target = reward + self._gamma * best_next_q
        self._q[state, action] += self._alpha * (td_target - self._q[state, action])
        self._episode_count += 1
        self.save()

    def decay_epsilon(self) -> None:
        """Multiply epsilon by epsilon_decay, floor at epsilon_min."""
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def reset(self) -> None:
        """Zero the Q-table and reset epsilon to initial value."""
        self._q = np.zeros((self._n_states, self._n_actions), dtype=np.float64)
        self._epsilon = self._epsilon_init
        self._episode_count = 0
        self.save()

    def save(self) -> None:
        """Persist Q-table as .npy file to qtable_path."""
        os.makedirs(os.path.dirname(self._qtable_path) or ".", exist_ok=True)
        np.save(self._qtable_path, self._q)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def top_actions(
        self,
        state: int,
        valid_actions: list[int],
        k: int = 3,
    ) -> list[dict]:
        """
        Return top-k actions sorted by Q-value descending.
        Each entry: { 'action': int, 'q_value': float, 'node_id': int }
        """
        scored = sorted(
            valid_actions,
            key=lambda a: self._q[state, a],
            reverse=True,
        )
        return [
            {"action": a, "q_value": float(self._q[state, a]), "node_id": a}
            for a in scored[:k]
        ]

    def confidence_spread(self, state: int, valid_actions: list[int]) -> float:
        """
        Return (best_q - second_best_q) / abs(best_q + 1e-9).
        Low spread means agent is uncertain — trigger Claude.
        """
        if len(valid_actions) < 2:
            return 1.0  # only one choice, fully confident
        sorted_q = sorted(
            (float(self._q[state, a]) for a in valid_actions), reverse=True
        )
        best_q = sorted_q[0]
        second_q = sorted_q[1]
        return (best_q - second_q) / abs(best_q + 1e-9)
