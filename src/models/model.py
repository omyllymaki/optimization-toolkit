from abc import abstractmethod, ABC
from typing import Tuple, Callable

import numpy as np


class Model(ABC):
    """
    Base model class for optimization.

    Inheritors need to implement update function that updates parameters that needs to be solved.
    """

    def __init__(self, f_eval: Callable, f_err: Callable, f_cost: Callable):
        """
        @param f_eval:  Function for evaluation: y_estimate = f_eval(x, param).
        @param f_err:  Function to calculate errors: errors = f_err(y_estimate, y).
        @param f_cost: Function to calculate cost: cost = f_cost(errors, param).
        """
        self.f_eval = f_eval
        self.f_err = f_err
        self.f_cost = f_cost

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
        y_eval = self.f_eval(x, param)
        return self.f_err(y_eval, y)

    def _cost(self, errors, param) -> float:
        return self.f_cost(errors, param)
