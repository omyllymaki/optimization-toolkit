import logging
from functools import partial
from typing import Tuple, Callable

import numpy as np
from numpy.linalg import pinv

from src.optimizer import Optimizer
from src.termination import check_n_iter, check_absolute_cost, check_n_iter_without_improvement
from src.utils import diff, gradient, mse

logger = logging.getLogger(__name__)

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=500),
    partial(check_absolute_cost, threshold=1e-6),
    partial(check_n_iter_without_improvement, threshold=20),
)


class LevenbergMarquardt(Optimizer):
    """
    Levenberg-Marquardt (LMA) optimization method.

    Levenberg-Marquardt is damped Gauss-Newton method where increment vector is rotated towards the direction of
    steepest descent. Amount of this rotation is determined by damping factor which is changed during iteration.
    If cost decreases, the parameters are updated and damping factor is decreased, making LMA to work more like
    standard Gauss-Newton. If cost increases, the parameters are not updated and damping factor is increased,
    making LMA to work more like gradient descent.
    """

    def __init__(self,
                 f_err: Callable,
                 damping_factor_scaling=2.0,
                 termination_checks=TERMINATION_CHECKS
                 ):
        """
        @param f_err: Function to calculate errors: errors = f_err(param). cost is then calculated as MSE(errors).
        @param damping_factor_scaling: Scale factor to change damping factor at every iteration (> 1).
        @param termination_checks: See Optimizer.
        """
        self.f_err = f_err
        self.weights = None
        self.damping_factor_scaling = damping_factor_scaling
        self.damping_factor = None
        super().__init__(self.f_cost, termination_checks)

    def update(self, param, iter_round, cost) -> Tuple[np.ndarray, float]:
        param_delta = self._calculate_update_delta(param)
        param_candidate = param - param_delta
        cost_candidate = self.f_cost(param_candidate)
        if cost_candidate < cost:
            param = param_candidate
            cost = cost_candidate
            self.damping_factor = self.damping_factor / self.damping_factor_scaling
            logger.debug(f"Cost decreased; decrease damping factor and update parameters")
        else:
            self.damping_factor = self.damping_factor_scaling * self.damping_factor
            logger.debug(f"Cost increased; increase damping factor and don't update parameters")
        logger.debug(f"Cost {cost:0.3f}; damping factor {self.damping_factor:0.3f}")

        return param, cost

    def _calculate_update_delta(self, param) -> np.ndarray:
        errors = self.f_err(param)
        jac = gradient(param, self.f_err)
        jj = jac.T @ jac
        if self.damping_factor is None:
            self.damping_factor = np.max(np.diag(jj))
        correction = np.diag(self.damping_factor * np.diag(jj))
        return pinv(jj + correction) @ jac.T @ errors

    def f_cost(self, param):
        return mse(self.f_err(param))
