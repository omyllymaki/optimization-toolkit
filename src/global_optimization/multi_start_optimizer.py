import logging
from collections import Callable, Iterable
from functools import partial
from typing import Tuple, Union

import numpy as np

from src.local_optimization.local_optimizer import LocalOptimizer
from src.termination import check_n_iter, check_n_iter_without_improvement, check_absolute_cost, check_termination

logger = logging.getLogger(__name__)

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=300),
    partial(check_n_iter_without_improvement, threshold=50),
    partial(check_absolute_cost, threshold=1e-6)
)


class MultiStartOptimizer:
    """
    Run specified local optimizer with multiple init guesses. New init guess is generated with f_init_guess function.
    Returns solution with smallest cost.
    """

    def __init__(self,
                 optimizer: LocalOptimizer,
                 f_init_guess: Callable,
                 termination_checks: Union[Tuple[Callable], Callable] = TERMINATION_CHECKS):
        """
        @param optimizer: local optimizer used for optimization.
        @param f_init_guess: Function to generate init guess: x0 = f_init_guess(xs, costs).
        @param termination_checks: Function or tuple of functions to check termination.
        """
        self.optimizer = optimizer
        self.f_init_guess = f_init_guess
        if not isinstance(termination_checks, Iterable):
            self.termination_checks = (termination_checks,)
        else:
            self.termination_checks = termination_checks

    def run(self):
        min_cost = np.inf
        x_output = None
        output_costs = []
        output_xs = None
        iter_round = 0
        while True:
            init_guess = self.f_init_guess(output_xs, output_costs)
            x, costs, _ = self.optimizer.run(init_guess)
            cost = np.min(costs)
            if cost < min_cost:
                x_output = x
            logger.info(f"Round {iter_round}: cost {cost:0.5f}")

            output_costs.append(cost)

            if output_xs is None:
                output_xs = x
            else:
                output_xs = np.vstack((output_xs, x))

            if check_termination(np.array(output_costs), self.termination_checks):
                break

            iter_round += 1

        return x_output, output_costs, output_xs
