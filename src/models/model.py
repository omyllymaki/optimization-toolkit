from abc import abstractmethod, ABC
from typing import Tuple, Callable

import numpy as np


class Model(ABC):
    """
    Base model class for optimization.

    Inheritors need to implement update function that updates parameters that needs to be solved.
    """

    def __init__(self, feval: Callable, ferr: Callable, fcost: Callable):
        """
        @param feval:  Function for evaluation: y_estimate = feval(x, param).
        @param ferr:  Function to calculate errors: errors = ferr(y_estimate, y).
        @param fcost: Function to calculate cost: cost = fcost(errors, param).
        """
        self.feval = feval
        self.ferr = ferr
        self.fcost = fcost

    @abstractmethod
    def update(self,
               param: np.ndarray,
               x: np.ndarray,
               y: np.ndarray,
               iter_round: int,
               cost: float) -> Tuple[np.ndarray, float]:
        """
        Update parameter that needs to be solved. Inheritors need to implement this.

        @param param: Current parameter values.
        @param x: Independent variables.
        @param y: Dependent variables.
        @param iter_round:  Current iteration round.
        @param cost: Current cost.

        @return Tuple containing updated parameters and new cost for updated parameters.
        """
        raise NotImplementedError

    def _errors(self, param, x, y) -> np.ndarray:
        y_eval = self.feval(x, param)
        return self.ferr(y_eval, y)

    def _cost(self, errors, param) -> float:
        return self.fcost(errors, param)
