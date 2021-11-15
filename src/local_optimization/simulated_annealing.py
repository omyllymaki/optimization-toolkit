import logging
import random
from functools import partial
from typing import Tuple, Callable

import numpy as np

from src.local_optimization.local_optimizer import LocalOptimizer
from src.termination import check_n_iter, check_n_iter_without_improvement, check_absolute_cost

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
    Calculate acceptance probability for x candidate, based on cost change (vs. current solution) and
    temperature.
    """
    if delta_cost < 0:
        return 1
    if temperature <= 0:
        return 0
    return np.exp(-delta_cost / temperature)


class SimulatedAnnealing(LocalOptimizer):
    """
    Simulated annealing optimizer.

    Generate new variables candidate. Replace the current variables with the candidate with probability that depends on
    cost difference and temperature.
    """

    def __init__(self,
                 f_update: Callable,
                 f_cost: Callable,
                 f_temp: Callable = temp_decay,
                 f_prob: Callable = acceptance_probability,
                 termination_checks=TERMINATION_CHECKS
                 ):
        """
        @param f_update: Function to generate x candidate: x_candidate = f_update(x, iter_round)
        @param f_cost: See LocalOptimizer.
        @param f_temp: Function to calculate current temperature from iteration round: temp = f_temp(iter_round)
        @param f_prob: Function to calculate acceptance probability for x candidate: prob = f_prob(delta_cost, temp)
        @param termination_checks: See LocalOptimizer.
        """
        super().__init__(f_cost, termination_checks)
        self.f_update = f_update
        self.f_temp = f_temp
        self.f_prob = f_prob

    def update(self, x, iter_round, cost) -> Tuple[np.ndarray, float]:
        temperature = self.f_temp(iter_round)
        x_candidate = self.f_update(x.copy(), iter_round)
        candidate_cost = self.f_cost(x_candidate)
        delta_cost = candidate_cost - cost
        prob = self.f_prob(delta_cost, temperature)
        logger.info(f"temp: {temperature} | delta cost {delta_cost} | prob {prob}")
        if prob >= random.uniform(0.0, 1.0):
            logger.info(f"Update solution!")
            return x_candidate, candidate_cost
        else:
            return x, cost
