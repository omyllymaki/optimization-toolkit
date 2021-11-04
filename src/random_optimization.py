from functools import partial
from typing import Tuple, Callable

import numpy as np

from src.optimizer import Optimizer
from src.termination import check_n_iter, check_n_iter_without_improvement, check_absolute_cost
from src.utils import mse

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=10000),
    partial(check_n_iter_without_improvement, threshold=2000),
    partial(check_absolute_cost, threshold=1e-6),
)


class RandomOptimization(Optimizer):
    """
    Random optimization optimizer.

    Create new parameter values candidate by adding a normally distributed random vector to the current parameter
    values. Update the current parameter values if cost of candidate is smaller than the current cost.
    """

    def __init__(self,
                 f_scaling: Callable,
                 f_cost: Callable,
                 termination_checks=TERMINATION_CHECKS,
                 ):
        """
        @param f_scaling: Function to scaling of parameter update: scale_factors = f_scaling(iter_round)
        @param f_cost: see Optimizer.
        @param termination_checks: See Optimizer.
        """
        super().__init__(f_cost, termination_checks)
        self.f_scaling = f_scaling

    def update(self, param, iter_round, cost) -> Tuple[np.ndarray, float]:
        scale_factors = self.f_scaling(iter_round)
        param_candidate = param + scale_factors * np.random.randn(param.shape[0])
        candidate_cost = self.f_cost(param_candidate)
        if candidate_cost < cost:
            param_out = param_candidate
            cost_out = candidate_cost
        else:
            param_out = param
            cost_out = cost
        return param_out, cost_out
