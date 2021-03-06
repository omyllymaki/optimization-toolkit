import logging
from functools import partial
from typing import Tuple, Callable

import numpy as np
from numpy.linalg import pinv

from src.local_optimization.gss import gss
from src.local_optimization.local_optimizer import LocalOptimizer
from src.termination import check_n_iter, check_absolute_cost, check_n_iter_without_improvement
from src.math import gradient
from src.loss import mse

logger = logging.getLogger(__name__)

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=500),
    partial(check_absolute_cost, threshold=1e-6),
    partial(check_n_iter_without_improvement, threshold=5)
)


class GaussNewton(LocalOptimizer):
    """
    Gauss-Newton optimizer.

    Minimize sum(errors^2) using Gauss-Newton (GN) method.

    Based on parameter choices, this method can be used as classical GN (step_size = 1) or as damped version (step size
    between 0 and 1). This method can also be used for iteratively re-weighted least squares where weights are updated
    every iteration, using specified function f_weights. This enables e.g. robust fitting.

    Note that Gauss-Newton optimizer takes f_err instead of f_cost as input. f_cost is then implicitly defined as

    f_cost = MSE(f_err(x))
    """

    def __init__(self,
                 f_err: Callable,
                 f_weights: Callable = None,
                 step_size_max_iter: int = 10,
                 step_size_lb: float = 0.0,
                 step_size_ub: float = 1.0,
                 termination_checks=TERMINATION_CHECKS
                 ):
        """
        @param f_err: Function to calculate errors: errors = f_err(x). cost is then calculated as MSE(errors).
        @param f_weights: Function to calculate weights for LS fit: weights = f_weights(errors)
        @param step_size_max_iter: Number of iterations for optimal step size search.
        @param step_size_lb: lower bound for step size.
        @param step_size_ub: Upper bound for step size.
        @param termination_checks: See LocalOptimizer.
        """
        self.f_err = f_err
        self.f_weights = f_weights
        self.step_size_max_iter = step_size_max_iter
        self.step_size_lb = step_size_lb
        self.step_size_ub = step_size_ub
        self.weights = None
        super().__init__(self.f_cost, termination_checks)

    def update(self, x, iter_round, cost) -> Tuple[np.ndarray, float]:
        x_delta = self._calculate_update_direction(x)
        step_size = self._find_step_size(x, x_delta)
        x = x - step_size * x_delta
        cost = self.f_cost(x)
        logger.debug(f"Cost {cost:0.3f}, step size {step_size:0.3f}")

        if self.f_weights is not None:
            errors = self.f_err(x)
            self.weights = self.f_weights(errors)

        return x, cost

    def _calculate_update_direction(self, x) -> np.ndarray:
        errors = self.f_err(x)
        jac = gradient(x, self.f_err)
        if self.weights is None:
            return pinv(jac.T @ jac) @ jac.T @ errors
        else:
            w = np.diag(self.weights)
            return pinv(jac.T @ w @ jac) @ jac.T @ w @ errors

    def _find_step_size(self, x, x_delta):
        if self.step_size_max_iter == 0:
            return (self.step_size_lb + self.step_size_ub) / 2
        f = partial(self._calculate_step_size_cost, x=x, x_delta=x_delta)
        d_min, d_max = gss(f, self.step_size_lb, self.step_size_ub, max_iter=self.step_size_max_iter)
        return (d_min + d_max) / 2

    def _calculate_step_size_cost(self, step_size, x, x_delta):
        x_candidate = x - step_size * x_delta
        return self.f_cost(x_candidate)

    def f_cost(self, x):
        if self.weights is None:
            return mse(self.f_err(x))
        else:
            return mse(self.weights * self.f_err(x))
