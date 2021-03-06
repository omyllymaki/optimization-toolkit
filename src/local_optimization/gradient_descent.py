import logging
from functools import partial
from typing import Tuple

import numpy as np

from src.local_optimization.gss import gss
from src.local_optimization.local_optimizer import LocalOptimizer
from src.termination import check_n_iter, check_absolute_cost, check_n_iter_without_improvement
from src.math import gradient

logger = logging.getLogger(__name__)

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=10000),
    partial(check_n_iter_without_improvement, threshold=1000),
    partial(check_absolute_cost, threshold=1e-6),
)


class GradientDescent(LocalOptimizer):
    """
    Gradient Descent optimizer.

    Minimize given cost function using Gradient descent method. Change step size adaptively for every iteration 
    according to given input parameters.
    """

    def __init__(self,
                 f_cost,
                 f_step=lambda _: (0, 1e-3),
                 step_size_max_iter=5,
                 termination_checks=TERMINATION_CHECKS
                 ):
        """
        @param f_cost: See LocalOptimizer.
        @param f_step: Function to calculate step size bounds for every iteration: lb, ub = f_step(iter_round)
        @param step_size_max_iter: Number of iterations for optimal step size search.
        @param termination_checks: See LocalOptimizer.
        """
        super().__init__(f_cost, termination_checks)
        self.f_step = f_step
        self.step_size_max_iter = step_size_max_iter
        self.step_size_lb = None
        self.step_size_ub = None

    def update(self, x, iter_round, cost) -> Tuple[np.ndarray, float]:
        self.step_size_lb, self.step_size_ub = self.f_step(iter_round)
        logger.debug(f"Step size range: [{self.step_size_lb}, {self.step_size_ub}]")
        x_delta = self._calculate_update_direction(x)
        step_size = self._find_step_size(x, x_delta)
        x = x - step_size * x_delta
        cost = self.f_cost(x)
        logger.debug(f"Cost {cost:0.3f}, step size {step_size}")
        return x, cost

    def _calculate_update_direction(self, x) -> np.ndarray:
        return gradient(x, self.f_cost)

    def _find_step_size(self, x, x_delta):
        if self.step_size_max_iter == 0:
            return (self.step_size_lb + self.step_size_ub) / 2
        f = partial(self._calculate_step_size_cost, x=x, x_delta=x_delta)
        d_min, d_max = gss(f, self.step_size_lb, self.step_size_ub, max_iter=self.step_size_max_iter)
        return (d_min + d_max) / 2

    def _calculate_step_size_cost(self, step_size, x, x_delta):
        x_candidate = x - step_size * x_delta
        return self.f_cost(x_candidate)
