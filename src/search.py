import logging
from collections import Callable, Iterable
from functools import partial
from typing import Tuple, Union

import numpy as np

from src.optimizer import Optimizer
from src.termination import check_n_iter, check_n_iter_without_improvement, check_absolute_cost, check_termination

logger = logging.getLogger(__name__)

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=300),
    partial(check_n_iter_without_improvement, threshold=50),
    partial(check_absolute_cost, threshold=1e-6)
)


def run_multiple(optimizer: Optimizer,
                 f_init_guess: Callable,
                 termination_checks: Union[Tuple[Callable], Callable] = TERMINATION_CHECKS):
    """
    Run optimizer with multiple init guesses. This might be useful e.g. when cost surface contains multiple local
    minima and there is a risk that optimizer will converge to local minimum that is not global minimum.

    @param optimizer: Optimizer used for optimization.
    @param f_init_guess: Function to generate new init guess for every iteration: init_guess = f_init_guess().
    @param termination_checks: Function or tuple of functions to check termination.

    @return: Tuple containing
    (parameters for best solution, costs from iteration, list of parameter values from iteration)

    """

    if not isinstance(termination_checks, Iterable):
        termination_checks = (termination_checks,)

    min_cost = np.inf
    output_param = None
    output_costs = []
    output_params = None
    while True:
        init_guess = f_init_guess()
        param, costs, _ = optimizer.run(init_guess)
        cost = np.min(costs)
        if cost < min_cost:
            output_param = param

        output_costs.append(cost)

        if output_params is None:
            output_params = param
        else:
            output_params = np.vstack((output_params, param))

        if check_termination(np.array(output_costs), termination_checks):
            break

    return output_param, output_costs, output_params
