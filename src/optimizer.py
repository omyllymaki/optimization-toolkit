import logging
from abc import abstractmethod, ABC
from collections import Iterable
from typing import Tuple, Callable, List, Union

import numpy as np

from src.termination import check_termination

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    """
    Base class for optimization.

    Inheritors need to implement update function that updates parameters that needs to be solved.
    """

    def __init__(self,
                 f_cost: Callable,
                 termination_checks: Union[Tuple[Callable], Callable]):
        """
        @param f_cost: Function to calculate cost: cost = f_cost(param).
        @param termination_checks: This if function or tuple of functions that return true if iteration should be
        terminated, otherwise false. All the functions will be checked and iteration is terminated if any of these
        return true.
        """
        self.f_cost = f_cost
        if not isinstance(termination_checks, Iterable):
            termination_checks = (termination_checks,)
        self.termination_checks = termination_checks

    def run(self,
            init_guess: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run optimization.

        @param init_guess: Initial guess for parameters.

        @return: Tuple containing
        (final solution for parameters, costs from iteration, list of parameter values from iteration)
        """
        param = init_guess.copy()
        final_param = init_guess.copy()
        cost = self.f_cost(param)
        min_cost = cost
        costs = [cost]
        params = param.copy()
        iter_round = 0
        logger.info(f"Init cost: {cost:0.5f}")
        while True:

            param, cost = self.update(param, iter_round, cost)
            costs.append(cost)
            params = np.vstack((params, param))
            logger.info(f"Round {iter_round}: cost {cost:0.5f}")
            iter_round += 1

            if cost < min_cost:
                min_cost = cost
                final_param = param.copy()

            if check_termination(np.array(costs), self.termination_checks):
                break

        return final_param, np.array(costs), params

    @abstractmethod
    def update(self,
               param: np.ndarray,
               iter_round: int,
               cost: float) -> Tuple[np.ndarray, float]:
        """
        Update parameter that needs to be solved. Inheritors need to implement this.

        @param param: Current parameter values.
        @param iter_round:  Current iteration round.
        @param cost: Current cost.

        @return Tuple containing updated parameters and new cost for updated parameters.
        """
        raise NotImplementedError
