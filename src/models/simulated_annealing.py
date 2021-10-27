import logging
import random
from typing import Tuple, Callable

import numpy as np

from src.models.model import Model
from src.utils import diff, rmse

logger = logging.getLogger(__name__)


def temp_decay(t: int, max_temperature=1.0, decay_constant=0.005) -> float:
    """
    Calculate current temperature from iteration round t.
    """
    return max_temperature * np.exp(-decay_constant * t)


def acceptance_probability(delta_cost: float, temperature: float) -> float:
    """
    Calculate acceptance probability for param candidate, based on cost change (vs. current solution) and
    temperature.
    """
    if delta_cost < 0:
        return 1
    else:
        return np.exp(-delta_cost / temperature)


class SimulatedAnnealing(Model):
    """
    Simulated annealing model.

    Generate new parameter candidate. Replace the current parameters with the candidate with probability that depends on
    cost difference and temperature.
    """

    def __init__(self,
                 f_eval: Callable,
                 f_update: Callable,
                 f_err: Callable = diff,
                 f_cost: Callable = rmse,
                 f_temp: Callable = temp_decay,
                 f_prob: Callable = acceptance_probability,
                 ):
        """
        @param f_eval: See Model.
        @param f_update: Function to generate param candidate: param_candidate = f_update(param, iter_round)
        @param f_err: See Model.
        @param f_cost: See Model.
        @param f_temp: Function to calculate current temperature from iteration round: temp = f_temp(iter_round)
        @param f_prob: Function to calculate acceptance probability for param candidate: prob = f_prob(delta_cost, temp)
        """
        super().__init__(f_eval, f_err, f_cost)
        self.f_update = f_update
        self.f_temp = f_temp
        self.f_prob = f_prob

    def update(self, param, x, y, iter_round, cost) -> Tuple[np.ndarray, float]:
        temperature = self.f_temp(iter_round)
        param_candidate = self.f_update(param.copy(), iter_round)
        errors = self._errors(param_candidate, x, y)
        param_candidate_cost = self.f_cost(errors, param_candidate)
        delta_cost = param_candidate_cost - cost
        prob = self.f_prob(delta_cost, temperature)
        logger.info(f"temp: {temperature} | delta cost {delta_cost} | prob {prob}")
        if prob >= random.uniform(0.0, 1.0):
            logger.info(f"Update solution!")
            return param_candidate, param_candidate_cost
        else:
            return param, cost
