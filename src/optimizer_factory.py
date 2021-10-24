from enum import Enum

from src.gauss_newton import GaussNewton
from src.gradient_descent import GradientDescent
from src.optimizer import Optimizer
from src.termination import TerminationCriteria

Method = Enum('Method', 'gn gd')


def get_optimizer(method: Method,
                  termination_criteria: TerminationCriteria = TerminationCriteria(),
                  *args, **kwargs):
    if method == Method.gn:
        model = GaussNewton(*args, **kwargs)
    elif method == Method.gd:
        model = GradientDescent(*args, **kwargs)
    else:
        raise Exception("Invalid method")
    return Optimizer(model=model, termination_criteria=termination_criteria)
