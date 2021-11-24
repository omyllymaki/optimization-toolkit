import logging
from abc import ABC

from src.optimization_results import Output

logger = logging.getLogger(__name__)


class GlobalOptimizer(ABC):
    """
    Base class for global optimization.

    Inheritors need to implement run function.
    """

    def run(self) -> Output:
        """
        Run global optimization.

        @return: optimization results.
        """
        raise NotImplementedError
