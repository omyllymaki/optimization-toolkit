from typing import Tuple, Callable

import numpy as np

from src.optimizer import Optimizer
from src.termination import TerminationCriteria as TC
from src.utils import diff, mse


class RandomOptimization(Optimizer):
    """
    Random optimization model.

    Create new parameter values candidate by adding a normally distributed random vector to the current parameter
    values. Update the current parameter values if cost of candidate is smaller than the current cost.
    """

    def __init__(self,
                 f_eval: Callable,
                 f_scaling: Callable,
                 f_err: Callable = diff,
                 f_cost: Callable = mse,
                 termination=TC(max_iter=10000,
                                max_iter_without_improvement=2000,
                                cost_threshold=1e-6,
                                cost_diff_threshold=-np.inf)
                 ):
        """
        @param f_eval: See Optimizer.
        @param f_scaling: Function to scaling of parameter update: scale_factors = f_scaling(iter_round)
        @param f_err: See Optimizer.
        @param f_cost: see Optimizer.
        @param termination: See Optimizer.
        """
        super().__init__(f_eval, f_err, f_cost, termination)
        self.f_scaling = f_scaling

    def update(self, param, x, y, iter_round, cost) -> Tuple[np.ndarray, float]:
        scale_factors = self.f_scaling(iter_round)
        param_candidate = param + scale_factors * np.random.randn(param.shape[0])
        errors = self._errors(param_candidate, x, y)
        candidate_cost = self._cost(errors, param_candidate)
        if candidate_cost < cost:
            param_out = param_candidate
            cost_out = candidate_cost
        else:
            param_out = param
            cost_out = cost
        return param_out, cost_out
