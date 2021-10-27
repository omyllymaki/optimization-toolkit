import logging
from functools import partial
from typing import Tuple

import numpy as np

from src.models.gss import gss
from src.models.model import Model
from src.utils import gradient, diff, mse

logger = logging.getLogger(__name__)


class GradientDescent(Model):
    """
    Gradient descent model.

    Minimize given cost function by optimizing parameters using Gradient descent method. Change step size adaptively
    for every iteration according to given input parameters.
    """

    def __init__(self,
                 fstep,
                 feval,
                 ferr=diff,
                 fcost=mse,
                 step_size_max_iter=5):
        """
        @param fstep: Function to calculate step size bounds for every iteration: lb, ub = fstep(iter_round)
        @param feval: See Model
        @param ferr: See Model.
        @param fcost: See Model.
        @param step_size_max_iter: Number of iterations for optimal step size search.
        """
        super().__init__(feval, ferr, fcost)
        self.fstep = fstep
        self.step_size_max_iter = step_size_max_iter
        self.step_size_lb = None
        self.step_size_ub = None

    def update(self, param, x, y, iter_round, cost) -> Tuple[np.ndarray, float]:
        self.step_size_lb, self.step_size_ub = self.fstep(iter_round)
        logger.debug(f"Step size range: [{self.step_size_lb}, {self.step_size_ub}]")
        param_delta = self._calculate_update_direction(param, x, y)
        step_size = self._find_step_size(param, x, y, param_delta)
        param = param - step_size * param_delta
        errors = self._errors(param, x, y)
        cost = self._cost(errors, param)
        logger.debug(f"Cost {cost:0.3f}, step size {step_size}")
        return param, cost

    def _calculate_update_direction(self, param, x, y) -> np.ndarray:
        f_cost = lambda p: self._cost(self._errors(p, x, y), p)  # cost, given just param as argument
        return gradient(param, f_cost)

    def _find_step_size(self, param, x, y, delta):
        if self.step_size_max_iter == 0:
            return (self.step_size_lb + self.step_size_ub) / 2
        f = partial(self._calculate_step_size_cost, param=param, delta=delta, x=x, y=y)
        d_min, d_max = gss(f, self.step_size_lb, self.step_size_ub, max_iter=self.step_size_max_iter)
        return (d_min + d_max) / 2

    def _calculate_step_size_cost(self, step_size, param, delta, x, y):
        param_candidate = param - step_size * delta
        errors = self._errors(param_candidate, x, y)
        return self._cost(errors, param_candidate)
