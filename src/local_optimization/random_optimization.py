from functools import partial
from typing import Tuple, Callable

import numpy as np

from src.local_optimization.local_optimizer import LocalOptimizer
from src.termination import check_n_iter, check_n_iter_without_improvement, check_absolute_cost

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=10000),
    partial(check_n_iter_without_improvement, threshold=2000),
    partial(check_absolute_cost, threshold=1e-6),
)


class RandomOptimization(LocalOptimizer):
    """
    Random optimization optimizer.

    Create new varibales candidate by adding a normally distributed random vector to the current variable
    values. Update the current variable values if cost of candidate is smaller than the current cost.
    """

    def __init__(self,
                 f_scaling: Callable,
                 f_cost: Callable,
                 termination_checks=TERMINATION_CHECKS,
                 ):
        """
        @param f_scaling: Function to calculate scaling for variables update: scale_factors = f_scaling(iter_round)
        @param f_cost: see LocalOptimizer.
        @param termination_checks: See LocalOptimizer.
        """
        super().__init__(f_cost, termination_checks)
        self.f_scaling = f_scaling

    def update(self, x, iter_round, cost) -> Tuple[np.ndarray, float]:
        scale_factors = self.f_scaling(iter_round)
        x_candidate = x + scale_factors * np.random.randn(x.shape[0])
        candidate_cost = self.f_cost(x_candidate)
        if candidate_cost < cost:
            x_out = x_candidate
            cost_out = candidate_cost
        else:
            x_out = x
            cost_out = cost
        return x_out, cost_out
