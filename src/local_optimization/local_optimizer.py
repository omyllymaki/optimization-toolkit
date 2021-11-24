import logging
from abc import abstractmethod, ABC
from collections import Iterable
from typing import Tuple, Callable, Union

import numpy as np

from src.optimization_results import Output
from src.termination import check_termination

logger = logging.getLogger(__name__)


class LocalOptimizer(ABC):
    """
    Base class for local optimization.

    Inheritors need to implement update function that updates variables x to minimize given cost function f_cost.
    """

    def __init__(self,
                 f_cost: Callable,
                 termination_checks: Union[Tuple[Callable], Callable]):
        """
        @param f_cost: Function to calculate cost: cost = f_cost(x).
        @param termination_checks: This if function or tuple of functions that return true if iteration should be
        terminated, otherwise false. All the functions will be checked and iteration is terminated if any of these
        return true.
        """
        self.f_cost = f_cost
        if not isinstance(termination_checks, Iterable):
            termination_checks = (termination_checks,)
        self.termination_checks = termination_checks

    def run(self, x0: np.ndarray) -> Output:
        """
        Run optimization.

        @param x0: Initial guess for variables.

        @return: optimization results.
        """
        x = x0.copy()
        final_solution = x0.copy()
        cost = self.f_cost(x)
        min_cost = cost
        costs = np.array([cost])
        xs = x.copy()
        iter_round = 0
        logger.info(f"Init cost: {cost:0.5f}")

        while True:
            x, cost = self.update(x, iter_round, cost)
            costs = np.append(costs, cost)
            xs = np.vstack((xs, x))
            logger.info(f"Round {iter_round}: cost {cost:0.5f}")
            iter_round += 1

            if cost < min_cost:
                min_cost = cost
                final_solution = x.copy()

            if check_termination(costs, self.termination_checks):
                break

        output = Output()
        output.x = final_solution
        output.min_cost = min_cost
        output.xs = xs
        output.costs = costs
        return output

    @abstractmethod
    def update(self,
               x: np.ndarray,
               iter_round: int,
               cost: float) -> Tuple[np.ndarray, float]:
        """
        Update variables that needs to be solved. Inheritors need to implement this.

        @param x: Current variable values.
        @param iter_round:  Current iteration round.
        @param cost: Current cost.

        @return Tuple containing updated variables and new cost for updated variables.
        """
        raise NotImplementedError
