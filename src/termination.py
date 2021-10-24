import logging

import numpy as np

logger = logging.getLogger(__name__)


class TerminationCriteria:

    def __init__(self,
                 min_iter: int = 1,
                 max_iter: int = 500,
                 cost_threshold: float = 1e-6,
                 cost_diff_threshold: float = 1e-9):
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.cost_threshold = cost_threshold
        self.cost_diff_threshold = cost_diff_threshold


def check_termination(costs: np.ndarray, parameters: TerminationCriteria) -> bool:
    k = len(costs)
    current_cost = costs[-1]
    if k > 1:
        diff = costs[-2] - costs[-1]
        if diff <= parameters.cost_diff_threshold and k >= parameters.min_iter:
            logger.info("Cost difference between iterations smaller than tolerance. Fit terminated.")
            return True
    if current_cost <= parameters.cost_threshold and k >= parameters.min_iter:
        logger.info("Cost smaller than tolerance. Fit terminated.")
        return True
    if k >= parameters.max_iter:
        logger.info("Max number of iterations reached. Fit didn't converge.")
        return True

    return False
