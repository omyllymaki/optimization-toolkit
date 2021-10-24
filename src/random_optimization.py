from typing import Tuple, Callable

import numpy as np

from src.model import Model
from src.utils import diff, rmse


class RandomOptimization(Model):

    def __init__(self,
                 feval,
                 fscaling: Callable,
                 ferr=diff,
                 fcost=rmse):
        super().__init__(feval, ferr, fcost)
        self.fscaling = fscaling

    def update(self, param, x, y, k, cost) -> Tuple[np.ndarray, float]:
        scale_factors = self.fscaling(k)
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
