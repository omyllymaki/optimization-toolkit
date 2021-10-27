import logging
from abc import abstractmethod, ABC
from typing import Tuple, Callable, List

import numpy as np

from src.termination import check_termination, TerminationCriteria

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    """
    Base class for optimization.

    Inheritors need to implement update function that updates parameters that needs to be solved.
    """

    def __init__(self,
                 f_eval: Callable,
                 f_err: Callable,
                 f_cost: Callable,
                 termination: TerminationCriteria):
        """
        @param f_eval: Function for evaluation: y_estimate = f_eval(x, param).
        @param f_err: Function to calculate errors: errors = f_err(y_estimate, y).
        @param f_cost: Function to calculate cost: cost = f_cost(errors, param).
        @param termination: Criteria for termination. Check all conditions and terminate if any of them is true.
        """
        self.f_eval = f_eval
        self.f_err = f_err
        self.f_cost = f_cost
        self.termination_criteria = termination

    def run(self,
            x: np.ndarray,
            y: np.ndarray,
            init_guess: np.ndarray) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Run optimization.

        @param x: Independent variables.
        @param y: Dependent variables.
        @param init_guess: Initial guess for parameters.

        @return: Tuple containing
        (final solution for parameters, costs from iteration, list of parameter values from iteration)
        """
        param = init_guess.copy()
        final_param = init_guess.copy()
        errors = self._errors(param, x, y)
        cost = self._cost(errors, param)
        min_cost = cost
        costs = [cost]
        params = [param]
        iter_round = 0
        while True:

            param, cost = self.update(param, x, y, iter_round, cost)
            costs.append(cost)
            params.append(param)
            logger.info(f"Round {iter_round}: cost {cost:0.5f}")
            iter_round += 1

            if cost < min_cost:
                min_cost = cost
                final_param = param.copy()

            if check_termination(np.array(costs), self.termination_criteria):
                break

        return final_param, costs, params

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
