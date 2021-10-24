import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


class TerminationCriteria:

    def __init__(self,
                 min_iter: int = 1,
                 max_iter: int = 500,
                 max_iter_without_improvement=None,
                 cost_threshold: float = 1e-6,
                 cost_diff_threshold: float = 1e-9):
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.max_iter_without_improvement = max_iter_without_improvement
        self.cost_threshold = cost_threshold
        self.cost_diff_threshold = cost_diff_threshold


def check_termination(costs: np.ndarray, parameters: TerminationCriteria) -> bool:
    n_rounds = len(costs)
    current_cost = costs[-1]
    if n_rounds > 1:
        diff = costs[-2] - costs[-1]
        if diff <= parameters.cost_diff_threshold and n_rounds >= parameters.min_iter:
            logger.info("Cost difference between iterations smaller than tolerance. Fit terminated.")
            return True
    if current_cost <= parameters.cost_threshold and n_rounds >= parameters.min_iter:
        logger.info("Cost smaller than tolerance. Fit terminated.")
        return True
    if n_rounds >= parameters.max_iter:
        logger.info("Max number of iterations reached.")
        return True
    if parameters.max_iter_without_improvement is not None and n_rounds >= parameters.max_iter_without_improvement:
        latest_costs = costs[-parameters.max_iter_without_improvement:]
        i_min = np.argmin(latest_costs)
        if i_min == 0:
            logger.info("Max number of iterations without improvement reached.")
            return True

    return False
