import logging
from functools import partial
from typing import Tuple, Callable

import numpy as np
from numpy.linalg import pinv

from src.gss import gss
from src.optimizer import Optimizer
from src.termination import check_n_iter, check_absolute_cost, check_absolute_cost_diff
from src.utils import gradient, mse

logger = logging.getLogger(__name__)

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=500),
    partial(check_absolute_cost, threshold=1e-6),
    partial(check_absolute_cost_diff, threshold=1e-9),
)


class GaussNewton(Optimizer):
    """
    Gauss-Newton optimizer.

    Minimize sum(errors^2) by optimizing parameters using Gauss-Newton (GN) method.

    Based on parameter choices, this method can be used as classical GN (step_size = 1) or as damped version (step size
    between 0 and 1). This method can also be used for iteratively re-weighted least squares where weights are updated
    every iteration, using specified function f_weights. This enables e.g. robust fitting.

    Note that Gauss-Newton optimizer takes f_err instead of f_cost as input. f_cost is then implicitly defined as

    f_cost = MSE(f_err(param))
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
        @param f_err: Function to calculate errors: errors = f_err(param). cost is then calculated as MSE(errors).
        @param f_weights: Function to calculate weights for LS fit: weights = f_weights(errors)
        @param step_size_max_iter: Number of iterations for optimal step size search.
        @param step_size_lb: lower bound for step size.
        @param step_size_ub: Upper bound for step size.
        @param step_size_ub: Upper bound for step size.
        @param termination_checks: See Optimizer.
        """
        self.f_err = f_err
        self.f_weights = f_weights
        self.step_size_max_iter = step_size_max_iter
        self.step_size_lb = step_size_lb
        self.step_size_ub = step_size_ub
        self.weights = None
        super().__init__(self.f_cost, termination_checks)

    def update(self, param, iter_round, cost) -> Tuple[np.ndarray, float]:
        param_delta = self._calculate_update_direction(param)
        step_size = self._find_step_size(param, param_delta)
        param = param - step_size * param_delta
        cost = self.f_cost(param)
        logger.debug(f"Cost {cost:0.3f}, step size {step_size:0.3f}")

        if self.f_weights is not None:
            errors = self.f_err(param)
            self.weights = self.f_weights(errors)

        return param, cost

    def _calculate_update_direction(self, param) -> np.ndarray:
        errors = self.f_err(param)
        jac = gradient(param, self.f_err)
        if self.weights is None:
            return pinv(jac.T @ jac) @ jac.T @ errors
        else:
            w = np.diag(self.weights)
            return pinv(jac.T @ w @ jac) @ jac.T @ w @ errors

    def _find_step_size(self, param, delta):
        if self.step_size_max_iter == 0:
            return (self.step_size_lb + self.step_size_ub) / 2
        f = partial(self._calculate_step_size_cost, param=param, delta=delta)
        d_min, d_max = gss(f, self.step_size_lb, self.step_size_ub, max_iter=self.step_size_max_iter)
        return (d_min + d_max) / 2

    def _calculate_step_size_cost(self, step_size, param, delta):
        param_candidate = param - step_size * delta
        return self.f_cost(param_candidate)

    def f_cost(self, param):
        if self.weights is None:
            return mse(self.f_err(param))
        else:
            return mse(self.weights * self.f_err(param))
