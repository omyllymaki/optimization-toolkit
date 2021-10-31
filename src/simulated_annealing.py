import logging
import random
from functools import partial
from typing import Tuple, Callable

import numpy as np

from src.optimizer import Optimizer
from src.termination import check_n_iter, check_n_iter_without_improvement, check_absolute_cost
from src.utils import mse

logger = logging.getLogger(__name__)

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=10000),
    partial(check_n_iter_without_improvement, threshold=2000),
    partial(check_absolute_cost, threshold=1e-6)
)


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
    if temperature <= 0:
        return 0
    return np.exp(-delta_cost / temperature)


class SimulatedAnnealing(Optimizer):
    """
    Simulated annealing optimizer.

    Generate new parameter candidate. Replace the current parameters with the candidate with probability that depends on
    cost difference and temperature.
    """

    def __init__(self,
                 f_update: Callable,
                 f_cost: Callable = mse,
                 f_temp: Callable = temp_decay,
                 f_prob: Callable = acceptance_probability,
                 termination_checks=TERMINATION_CHECKS
                 ):
        """
        @param f_update: Function to generate param candidate: param_candidate = f_update(param, iter_round)
        @param f_cost: See Optimizer.
        @param f_temp: Function to calculate current temperature from iteration round: temp = f_temp(iter_round)
        @param f_prob: Function to calculate acceptance probability for param candidate: prob = f_prob(delta_cost, temp)
        @param termination_checks: See Optimizer.
        """
        super().__init__(f_cost, termination_checks)
        self.f_update = f_update
        self.f_temp = f_temp
        self.f_prob = f_prob

    def update(self, param, iter_round, cost) -> Tuple[np.ndarray, float]:
        temperature = self.f_temp(iter_round)
        param_candidate = self.f_update(param.copy(), iter_round)
        param_candidate_cost = self.f_cost(param_candidate)
        delta_cost = param_candidate_cost - cost
        prob = self.f_prob(delta_cost, temperature)
        logger.info(f"temp: {temperature} | delta cost {delta_cost} | prob {prob}")
        if prob >= random.uniform(0.0, 1.0):
            logger.info(f"Update solution!")
            return param_candidate, param_candidate_cost
        else:
            return param, cost
