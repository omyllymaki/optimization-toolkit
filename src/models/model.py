from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np


class Model(ABC):

    def __init__(self, feval, ferr, fcost):
        self.feval = feval
        self.ferr = ferr
        self.fcost = fcost

    @abstractmethod
    def update(self, param, x, y, iteration_round, cost) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def _errors(self, param, x, y) -> np.ndarray:
        y_eval = self.feval(x, param)
        return self.ferr(y_eval, y)

    def _cost(self, errors):
        return self.fcost(errors)
