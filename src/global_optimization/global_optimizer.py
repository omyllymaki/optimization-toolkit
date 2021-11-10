import logging
from abc import ABC
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GlobalOptimizer(ABC):
    """
    Base class for global optimization.

    Inheritors need to implement run function.
    """

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run global optimization.

        @return: Tuple containing
        (final solution for parameters, costs from iteration, list of parameter values from iteration)
        """
        raise NotImplementedError
