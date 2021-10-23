import logging
from abc import abstractmethod
from functools import partial
from typing import Tuple

import numpy as np

from src.gss import gss
from src.model import Model
from src.utils import diff

logger = logging.getLogger(__name__)


class BaseModel(Model):

    def __init__(self,
                 feval,
                 ferr=diff,
                 cost_diff_tol=1e-6,
                 cost_tol=1e-6,
                 df_search_max_iter=10,
                 min_iter=5,
                 max_iter=100,
                 df_min=0.0,
                 df_max=1.0):
        super().__init__(feval, ferr)
        self.cost_diff_tol = cost_diff_tol
        self.cost_tol = cost_tol
        self.df_search_max_iter = df_search_max_iter
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.df_min = df_min
        self.df_max = df_max

    @abstractmethod
    def _calculate_update_direction(self, param, x, y) -> np.ndarray:
        raise NotImplementedError

    def update(self, param, x, y) -> Tuple[np.ndarray, float]:

        # Find direction for param update
        delta = self._calculate_update_direction(param, x, y)

        # Find damping factor (step size) that approximately minimizes cost in determined direction
        damping_factor = self._find_damping_factor(param, x, y, delta)

        # Update coefficients
        param = param - damping_factor * delta

        # Calculate cost
        errors = self._calculate_errors(param, x, y)
        cost = self._calculate_cost(errors)
        logger.debug(f"Cost {cost:0.3f}, damping factor {damping_factor:0.3f}")

        return param, cost

    def check_termination(self, costs: np.ndarray) -> bool:
        k = len(costs)
        current_cost = costs[-1]
        if k > 1:
            diff = costs[-2] - costs[-1]
            if diff <= self.cost_diff_tol and k >= self.min_iter:
                logger.info("Cost difference between iterations smaller than tolerance. Fit terminated.")
                return True
        if current_cost <= self.cost_tol and k >= self.min_iter:
            logger.info("Cost smaller than tolerance. Fit terminated.")
            return True
        if k >= self.max_iter:
            logger.info("Max number of iterations reached. Fit didn't converge.")
            return True

        return False

    def _calculate_errors(self, param, x, y) -> np.ndarray:
        y_eval = self.feval(x, param)
        return self.ferr(y_eval, y)

    def _find_damping_factor(self, param, x, y, delta):
        f = partial(self._calculate_damping_factor_cost, param=param, delta=delta, x=x, y=y)
        d_min, d_max = gss(f, self.df_min, self.df_max, max_iter=self.df_search_max_iter)
        return (d_min + d_max) / 2

    def _calculate_damping_factor_cost(self, damping_factor, param, delta, x, y):
        param_candidate = param - damping_factor * delta
        errors = self._calculate_errors(param_candidate, x, y)
        return self._calculate_cost(errors)

    @staticmethod
    def _calculate_cost(errors):
        return np.sqrt(np.sum(errors ** 2))
