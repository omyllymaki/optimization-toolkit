import logging
from functools import partial
from typing import Tuple, Callable, List

import numpy as np
from numpy.linalg import pinv

from src.gss import gss
from src.optimizer import Optimizer
from src.termination import TerminationCriteria as TC
from src.utils import diff, gradient, mse

logger = logging.getLogger(__name__)


class GaussNewton(Optimizer):
    """
    Gauss-Newton model.

    Minimize sum(residual^2) by optimizing parameters using Gauss-Newton (GN) method.

    Based on parameter choices, this model can be used as classical GN (step_size = 1) or as damped version (step size
    between 0 and 1). This model can also be used for iteratively re-weighted least squares where weights are updated
    every iteration, using specified function f_weights. This enables e.g. robust fitting.
    """

    def __init__(self,
                 f_eval: Callable,
                 f_err: Callable = diff,
                 f_cost: Callable = mse,
                 f_weights: Callable = None,
                 step_size_max_iter: int = 10,
                 step_size_lb: float = 0.0,
                 step_size_ub: float = 1.0,
                 termination=TC(max_iter=500,
                                cost_threshold=1e-6,
                                cost_diff_threshold=1e-9)
                 ):
        """
        @param f_eval: See Optimizer.
        @param f_err: See Optimizer.
        @param f_cost: See Optimizer.
        @param f_weights: Function to calculate weights for LS fit: weights = f_weights(errors)
        @param step_size_max_iter: Number of iterations for optimal step size search.
        @param step_size_lb: lower bound for step size.
        @param step_size_ub: Upper bound for step size.
        @param step_size_ub: Upper bound for step size.
        @param termination: See Optimizer.
        """
        super().__init__(f_eval, f_err, f_cost, termination)
        self.f_weights = f_weights
        self.step_size_max_iter = step_size_max_iter
        self.step_size_lb = step_size_lb
        self.step_size_ub = step_size_ub
        self.weights = None

    def update(self, param, x, y, iter_round, cost) -> Tuple[np.ndarray, float]:
        if iter_round == 0:
            self.weights = np.ones(y.shape[0])

        param_delta = self._calculate_update_direction(param, x, y)
        step_size = self._find_step_size(param, x, y, param_delta)
        param = param - step_size * param_delta
        errors = self._errors(param, x, y)
        cost = self._cost(self.weights * errors, param)
        logger.debug(f"Cost {cost:0.3f}, step size {step_size:0.3f}")

        if self.f_weights is not None:
            self.weights = self.f_weights(errors)

        return param, cost

    def _calculate_update_direction(self, param, x, y) -> np.ndarray:
        errors = self._errors(param, x, y)
        f = partial(self._errors, x=x, y=y)
        jac = gradient(param, f)
        w = np.diag(self.weights)
        return pinv(jac.T @ w @ jac) @ jac.T @ w @ errors

    def _find_step_size(self, param, x, y, delta):
        if self.step_size_max_iter == 0:
            return (self.step_size_lb + self.step_size_ub) / 2
        f = partial(self._calculate_step_size_cost, param=param, delta=delta, x=x, y=y)
        d_min, d_max = gss(f, self.step_size_lb, self.step_size_ub, max_iter=self.step_size_max_iter)
        return (d_min + d_max) / 2

    def _calculate_step_size_cost(self, step_size, param, delta, x, y):
        param_candidate = param - step_size * delta
        errors = self._errors(param_candidate, x, y)
        return self._cost(self.weights * errors, param_candidate)
