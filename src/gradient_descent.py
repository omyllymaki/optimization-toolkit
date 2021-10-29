import logging
from functools import partial
from typing import Tuple

import numpy as np

from src.gss import gss
from src.optimizer import Optimizer
from src.termination import TerminationCriteria as TC
from src.utils import gradient

logger = logging.getLogger(__name__)


class GradientDescent(Optimizer):
    """
    Gradient descent optimizer.

    Minimize given cost function by optimizing parameters using Gradient descent method. Change step size adaptively
    for every iteration according to given input parameters.
    """

    def __init__(self,
                 f_step,
                 f_cost,
                 step_size_max_iter=5,
                 termination=TC(max_iter=10000,
                                max_iter_without_improvement=1000,
                                cost_threshold=1e-6,
                                cost_diff_threshold=-np.inf)
                 ):
        """
        @param f_step: Function to calculate step size bounds for every iteration: lb, ub = f_step(iter_round)
        @param f_cost: See Optimizer.
        @param step_size_max_iter: Number of iterations for optimal step size search.
        @param termination: See Optimizer.
        """
        super().__init__(f_cost, termination)
        self.f_step = f_step
        self.step_size_max_iter = step_size_max_iter
        self.step_size_lb = None
        self.step_size_ub = None

    def update(self, param, iter_round, cost) -> Tuple[np.ndarray, float]:
        self.step_size_lb, self.step_size_ub = self.f_step(iter_round)
        logger.debug(f"Step size range: [{self.step_size_lb}, {self.step_size_ub}]")
        param_delta = self._calculate_update_direction(param)
        step_size = self._find_step_size(param, param_delta)
        param = param - step_size * param_delta
        cost = self.f_cost(param)
        logger.debug(f"Cost {cost:0.3f}, step size {step_size}")
        return param, cost

    def _calculate_update_direction(self, param) -> np.ndarray:
        return gradient(param, self.f_cost)

    def _find_step_size(self, param, delta):
        if self.step_size_max_iter == 0:
            return (self.step_size_lb + self.step_size_ub) / 2
        f = partial(self._calculate_step_size_cost, param=param, delta=delta)
        d_min, d_max = gss(f, self.step_size_lb, self.step_size_ub, max_iter=self.step_size_max_iter)
        return (d_min + d_max) / 2

    def _calculate_step_size_cost(self, step_size, param, delta):
        param_candidate = param - step_size * delta
        return self.f_cost(param_candidate)
