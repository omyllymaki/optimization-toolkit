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
        param = init_guess
        final_param = init_guess
        costs, params = [], []
        min_cost = np.inf
        round_counter = 0
        cost = np.inf
        while True:

            param, cost = self.model.update(param, x, y, round_counter, cost)
            costs.append(cost)
            params.append(param)
            logger.info(f"Round {round_counter}: cost {cost:0.5f}")
            round_counter += 1

            if cost < min_cost:
                min_cost = cost
                final_param = param.copy()

            if check_termination(np.array(costs), self.termination_criteria):
                break

        return final_param, costs, params
