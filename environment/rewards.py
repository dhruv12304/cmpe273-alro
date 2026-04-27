import os

REWARD_ON_TIME = float(os.getenv("REWARD_ON_TIME", 100.0))
PENALTY_PER_STEP = float(os.getenv("PENALTY_PER_STEP", -1.0))
PENALTY_LATE = float(os.getenv("PENALTY_LATE", -20.0))
PENALTY_NO_STOCK = float(os.getenv("PENALTY_NO_STOCK", -30.0))


def calculate_reward(
    on_time: bool,
    steps_taken: int,
    deadline: int,
    invalid_warehouse: bool,
) -> float:
    """Pure function. Returns composite reward for a terminal step."""
    r = PENALTY_PER_STEP
    if on_time:
        r += REWARD_ON_TIME
    else:
        r += PENALTY_LATE
    if invalid_warehouse:
        r += PENALTY_NO_STOCK
    return r
