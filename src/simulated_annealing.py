import logging
import random
from typing import Tuple

import numpy as np

from src.model import Model
from src.utils import diff, rmse

logger = logging.getLogger(__name__)


class SimulatedAnnealing(Model):

    def __init__(self,
                 feval,
                 fupdate,
                 ferr=diff,
                 fcost=rmse,
                 max_temperature=1.0,
                 decay_constant=0.005,
                 ):
        super().__init__(feval, ferr, fcost)
        self.fupdate = fupdate
        self.max_temperature = max_temperature
        self.decay_constant = decay_constant

    def update(self, param, x, y, iteration_round, cost) -> Tuple[np.ndarray, float]:
        temperature = self._schedule(iteration_round)
        param_candidate = self.fupdate(param.copy(), iteration_round)
        errors = self._errors(param_candidate, x, y)
        param_candidate_cost = self.fcost(errors)
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
