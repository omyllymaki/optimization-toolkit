import logging
from typing import Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)


def check_termination(costs: np.ndarray, checks: Tuple[Callable]) -> bool:
    for check in checks:
        if check(costs):
            return True
    return False


def check_n_iter(costs: np.ndarray, threshold: int) -> bool:
    if len(costs) >= threshold:
        logger.info(f"Max number of iterations reached: {threshold}")
        return True
    else:
        return False


def check_absolute_cost(costs: np.ndarray, threshold: float) -> bool:
    if costs[-1] <= threshold:
        logger.info(f"Cost is smaller than or equal to threshold: {costs[-1]} <= {threshold}")
        return True
    else:
        return False


def check_absolute_cost_diff(costs: np.ndarray, threshold: float, n: int = 2) -> bool:
    if len(costs) < n:
        return False
    diff = costs[-n] - costs[-1]
    if diff <= threshold:
        logger.info(f"Cost difference is smaller than or equal to threshold: {diff} <= {threshold}")
        return True
    else:
        return False


def check_relative_cost_diff(costs: np.ndarray, threshold: float, n: int = 2) -> bool:
    if len(costs) < n:
        return False
    rel_diff = (costs[-n] - costs[-1]) / costs[-1]
    if rel_diff <= threshold:
        logger.info(f"Relative cost difference is smaller than or equal to threshold: {rel_diff} <= {threshold}")
        return True
    else:
        return False


def check_n_iter_without_improvement(costs: np.ndarray, threshold: int) -> bool:
    if len(costs) < threshold:
        return False
    latest_costs = costs[-threshold:]
    i_min = np.argmin(latest_costs)
    if i_min == 0:
        logger.info(f"Max number of iterations without improvement reached: {threshold}")
        return True
    else:
        return False
