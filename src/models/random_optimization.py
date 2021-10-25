from typing import Tuple, Callable

import numpy as np

from src.models.model import Model
from src.utils import diff, rmse


class RandomOptimization(Model):
    """
    Random optimization model.

    Create new parameter values candidate by adding a normally distributed random vector to the current parameter
    values. Update the current parameter values if cost of candidate is smaller than the current cost.
    """

    def __init__(self,
                 feval: Callable,
                 fscaling: Callable,
                 ferr: Callable = diff,
                 fcost: Callable = rmse):
        """
        @param feval: See Model.
        @param fscaling: Function to scaling of parameter update: scale_factors = fscaling(iter_round)
        @param ferr: See Model.
        @param fcost: see Model.
        """
        super().__init__(feval, ferr, fcost)
        self.fscaling = fscaling

    def update(self, param, x, y, iteration_round, cost) -> Tuple[np.ndarray, float]:
        scale_factors = self.fscaling(iteration_round)
        param_candidate = param + scale_factors * np.random.randn(param.shape[0])
        errors = self._errors(param_candidate, x, y)
        candidate_cost = self._cost(errors)
        if candidate_cost < cost:
            param_out = param_candidate
            cost_out = candidate_cost
        else:
            param_out = param
            cost_out = cost
        return param_out, cost_out
