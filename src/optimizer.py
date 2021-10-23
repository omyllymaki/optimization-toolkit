import logging

import numpy as np

logger = logging.getLogger(__name__)


class Optimizer:

    def __init__(self, model):
        self.model = model

    def fit(self, x, y, init_guess):

        param = init_guess
        final_param = init_guess
        costs, params = [], []
        min_cost = np.inf
        round_counter = 1
        while True:

            param, cost = self.model.update(param, x, y)
            costs.append(cost)
            params.append(param)
            logger.info(f"Round {round_counter}: cost {cost:0.5f}")
            round_counter += 1

            if cost < min_cost:
                min_cost = cost
                final_param = param.copy()

            if self.model.check_termination(costs):
                break

        return final_param, costs, params
