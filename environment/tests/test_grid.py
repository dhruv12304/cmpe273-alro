"""Acceptance tests for the Grid Environment library (Component A)."""

import random
import pytest

from environment.grid import LogisticsGrid
from environment.baseline import greedy_route
from environment.rewards import REWARD_ON_TIME, PENALTY_PER_STEP


# ---------------------------------------------------------------------------
# test_reset_deterministic
# ---------------------------------------------------------------------------

def test_reset_deterministic():
    """Same seed → same congestion weights and source/destination on every reset()."""
    grid = LogisticsGrid(n=5)

    obs1 = grid.reset(seed=42)
    cong1 = dict(obs1["congestion"])
    src1 = obs1["source"]
    dst1 = obs1["destination"]

    obs2 = grid.reset(seed=42)
    cong2 = dict(obs2["congestion"])
    src2 = obs2["source"]
    dst2 = obs2["destination"]

    assert src1 == src2
    assert dst1 == dst2
    assert cong1 == cong2


# ---------------------------------------------------------------------------
# test_step_invalid_action
# ---------------------------------------------------------------------------

def test_step_invalid_action():
    """step() with a non-adjacent node index raises ValueError."""
    grid = LogisticsGrid(n=5)
    grid.reset(seed=0)

    # Find a node index that is definitely NOT adjacent to the current node
    state = grid.encode_state(grid._current_node, grid._destination_node)
    valid = set(grid.valid_actions(state))
    # Pick any node not in valid and not the current node itself
    invalid_action = next(
        n for n in range(grid._n_nodes)
        if n not in valid and n != grid._current_node
    )

    with pytest.raises(ValueError):
        grid.step(invalid_action)


# ---------------------------------------------------------------------------
# test_episode_terminates
# ---------------------------------------------------------------------------

def test_episode_terminates():
    """Running valid_actions() → step() in a loop always reaches done=True within n*n steps."""
    grid = LogisticsGrid(n=5)
    grid.reset(seed=7)

    max_steps = grid._n * grid._n
    done = False
    for _ in range(max_steps + 1):
        state = grid.encode_state(grid._current_node, grid._destination_node)
        actions = grid.valid_actions(state)
        action = random.choice(actions)
        _, _, done, _ = grid.step(action)
        if done:
            break

    assert done, "Episode did not terminate within n*n steps"


# ---------------------------------------------------------------------------
# test_reward_on_time
# ---------------------------------------------------------------------------

def test_reward_on_time():
    """
    Reaching destination within deadline returns cumulative reward >=
    REWARD_ON_TIME + (steps * PENALTY_PER_STEP).
    """
    # Use a 2×2 grid so reaching destination is trivial (1–2 steps)
    grid = LogisticsGrid(n=2)
    # origin=0 (warehouse, col 0), destination=1 (destination, col 1 = n-1=1)
    grid.reset(seed=1, origin=0, destination=1)

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        state = grid.encode_state(grid._current_node, grid._destination_node)
        actions = grid.valid_actions(state)
        # Greedily pick destination if reachable, else any action
        action = grid._destination_node if grid._destination_node in actions else actions[0]
        _, reward, done, info = grid.step(action)
        total_reward += reward
        steps += 1

    assert info["on_time"], "Should have delivered on time on short grid"
    expected_min = REWARD_ON_TIME + steps * PENALTY_PER_STEP
    assert total_reward >= expected_min, (
        f"Total reward {total_reward} < expected minimum {expected_min}"
    )


# ---------------------------------------------------------------------------
# test_greedy_vs_random
# ---------------------------------------------------------------------------

def test_greedy_vs_random():
    """greedy_route() produces lower average cost than random policy over 100 episodes."""
    grid = LogisticsGrid(n=5)

    greedy_costs = []
    random_costs = []

    for ep in range(100):
        # --- greedy ---
        obs = grid.reset(seed=ep)
        order = {
            "origin": obs["source"],
            "destination": obs["destination"],
            "deadline": grid._n * 2,
            "quantity": 0,
        }
        result = greedy_route(grid, order)
        greedy_costs.append(result["cost"])

        # --- random ---
        grid.reset(seed=ep)
        total_cost = 0.0
        done = False
        for _ in range(grid.n_states):
            state = grid.encode_state(grid._current_node, grid._destination_node)
            actions = grid.valid_actions(state)
            action = random.choice(actions)
            _, _, done, info = grid.step(action)
            total_cost += info["cost"]
            if done:
                break
        random_costs.append(total_cost)

    avg_greedy = sum(greedy_costs) / len(greedy_costs)
    avg_random = sum(random_costs) / len(random_costs)

    assert avg_greedy < avg_random, (
        f"Greedy avg cost {avg_greedy:.2f} not less than random avg cost {avg_random:.2f}"
    )
