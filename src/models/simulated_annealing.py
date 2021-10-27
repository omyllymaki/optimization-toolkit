import logging
import random
from typing import Tuple, Callable

import numpy as np

from src.models.model import Model
from src.utils import diff, rmse

logger = logging.getLogger(__name__)


class SimulatedAnnealing(Model):
    """
    Simulated annealing model.

    Generate new parameter candidate. Replace the current parameters with the candidate with probability that depends on
    cost difference and temperature.
    """

    def __init__(self,
                 feval: Callable,
                 fupdate: Callable,
                 ferr: Callable = diff,
                 fcost: Callable = rmse,
                 max_temperature: float = 1.0,
                 decay_constant: float = 0.005,
                 ):
        """
        @param feval: See Model.
        @param fupdate: Function to generate param candidate: param_candidate = fupdate(param, iter_round)
        @param ferr: See Model.
        @param fcost: See Model.
        @param max_temperature: Maximum temperature.
        @param decay_constant: Decay constant for temperature.
        """
        super().__init__(feval, ferr, fcost)
        self.fupdate = fupdate
        self.max_temperature = max_temperature
        self.decay_constant = decay_constant

    def update(self, param, x, y, iter_round, cost) -> Tuple[np.ndarray, float]:
        temperature = self._schedule(iter_round)
        param_candidate = self.fupdate(param.copy(), iter_round)
        errors = self._errors(param_candidate, x, y)
        param_candidate_cost = self.fcost(errors, param_candidate)
        delta_cost = param_candidate_cost - cost
        prob = self._probability(delta_cost, temperature)
        logger.info(f"temp: {temperature} | delta cost {delta_cost} | prob {prob}")
        if prob >= random.uniform(0.0, 1.0):
            logger.info(f"Update solution!")
            return param_candidate, param_candidate_cost
        else:
            return param, cost

    def _schedule(self, t: int) -> float:
        """
        Calculate current temperature from iteration round t.
        """
        return self.max_temperature * np.exp(-self.decay_constant * t)

    @staticmethod
    def _probability(delta_cost: float, temperature: float) -> float:
        """
        Calculate acceptance probability for param candidate, based on cost change (vs. current solution) and
        temperature.
        """
        if delta_cost < 0:
            return 1
        else:
            return np.exp(-delta_cost / temperature)
