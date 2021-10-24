from enum import Enum

from src.models.gauss_newton import GaussNewton
from src.models.gradient_descent import GradientDescent
from src.optimizer import Optimizer
from src.models.random_optimization import RandomOptimization
from src.models.simulated_annealing import SimulatedAnnealing
from src.termination import TerminationCriteria

Method = Enum('Method', 'gn gd ro sa')


def get_optimizer(method: Method,
                  termination_criteria: TerminationCriteria = TerminationCriteria(),
                  *args, **kwargs):
    if method == Method.gn:
        model = GaussNewton(*args, **kwargs)
    elif method == Method.gd:
        model = GradientDescent(*args, **kwargs)
    elif method == Method.ro:
        model = RandomOptimization(*args, **kwargs)
    elif method == Method.sa:
        model = SimulatedAnnealing(*args, **kwargs)
    else:
        raise Exception("Invalid method")
    return Optimizer(model=model, termination_criteria=termination_criteria)
