import logging

import numpy as np

from src.model import Model
from src.termination import TerminationCriteria, check_termination

logger = logging.getLogger(__name__)


class Optimizer:

    def __init__(self,
                 model: Model,
                 termination_criteria: TerminationCriteria = TerminationCriteria()):
        self.model = model
        self.termination_criteria = termination_criteria

    def fit(self, x, y, init_guess):

        param = init_guess
        final_param = init_guess
        costs, params = [], []
        min_cost = np.inf
        round_counter = 0
        while True:

            param, cost = self.model.update(param, x, y, round_counter)
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
