import logging
from functools import partial
from typing import Tuple

import numpy as np

from src.models.gss import gss
from src.models.model import Model
from src.utils import diff, gradient, pseudoinverse, mse

logger = logging.getLogger(__name__)


class GaussNewton(Model):

    def __init__(self,
                 feval,
                 ferr=diff,
                 fcost=mse,
                 df_search_max_iter=10,
                 df_min=0.0,
                 df_max=1.0):
        super().__init__(feval, ferr, fcost)
        self.df_search_max_iter = df_search_max_iter
        self.df_min = df_min
        self.df_max = df_max

    def update(self, param, x, y, iteration_round, cost) -> Tuple[np.ndarray, float]:
        param_delta = self._calculate_update_direction(param, x, y)
        step_size = self._find_step_size(param, x, y, param_delta)
        param = param - step_size * param_delta
        errors = self._errors(param, x, y)
        cost = self._cost(errors)
        logger.debug(f"Cost {cost:0.3f}, step size {step_size:0.3f}")
        return param, cost

    def _calculate_update_direction(self, param, x, y) -> np.ndarray:
        errors = self._errors(param, x, y)
        f = partial(self._errors, x=x, y=y)
        jacobian = gradient(param, f)
        return pseudoinverse(jacobian) @ errors

    def _find_step_size(self, param, x, y, delta):
        if self.df_search_max_iter == 0:
            return (self.df_min + self.df_max) / 2
        f = partial(self._calculate_damping_factor_cost, param=param, delta=delta, x=x, y=y)
        d_min, d_max = gss(f, self.df_min, self.df_max, max_iter=self.df_search_max_iter)
        return (d_min + d_max) / 2

    def _calculate_damping_factor_cost(self, damping_factor, param, delta, x, y):
        param_candidate = param - damping_factor * delta
        errors = self._errors(param_candidate, x, y)
        return self._cost(errors)
