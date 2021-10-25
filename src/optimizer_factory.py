from enum import Enum

from src.models.gauss_newton import GaussNewton
from src.models.gradient_descent import GradientDescent
from src.models.random_optimization import RandomOptimization
from src.models.simulated_annealing import SimulatedAnnealing
from src.optimizer import Optimizer
from src.termination import TerminationCriteria


class Method(Enum):
    GN = GaussNewton
    GD = GradientDescent
    RO = RandomOptimization
    SA = SimulatedAnnealing


def get_optimizer(method: Method,
                  termination_criteria: TerminationCriteria = TerminationCriteria(),
                  *args, **kwargs):
    model = method.value(*args, **kwargs)
    return Optimizer(model=model, termination_criteria=termination_criteria)
