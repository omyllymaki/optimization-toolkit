import logging
from typing import Tuple, List

import numpy as np

from src.models.model import Model
from src.termination import TerminationCriteria, check_termination

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Optimizer that does iterative optimization of parameters, given a model for parameter update.
    """

    def __init__(self,
                 model: Model,
                 termination_criteria: TerminationCriteria = TerminationCriteria()):
        """
        @param model: Model for parameter update.
        @param termination_criteria: Criteria for termination of optimization.
        """
        self.model = model
        self.termination_criteria = termination_criteria

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            init_guess: np.ndarray) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Fit model

        @param x: Independent variables.
        @param y: Dependent variables.
        @param init_guess: Initial guess for parameters.

        @return: Tuple containing
        (final solution for parameters, costs from iteration, list of parameter values from iteration)
        """
        param = init_guess.copy()
        final_param = init_guess.copy()
        errors = self.model._errors(param, x, y)
        cost = self.model._cost(errors, param)
        min_cost = cost
        costs = [cost]
        params = [param]
        iter_round = 0
        while True:

            param, cost = self.model.update(param, x, y, iter_round, cost)
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
