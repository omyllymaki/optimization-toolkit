from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np

from src.utils import diff


class Model(ABC):

    def __init__(self, feval, ferr=diff):
        self.feval = feval
        self.ferr = ferr

    @abstractmethod
    def update(self, param, x, y, k) -> Tuple[np.ndarray, float]:
        raise NotImplementedError
