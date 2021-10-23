from enum import Enum

from src.damped_gn import DampedGN
from src.gradient_descent import GD
from src.optimizer import Optimizer

Method = Enum('Method', 'dgn gd')


def get_optimizer(method: Method, *args, **kwargs):
    if method == Method.dgn:
        model = DampedGN(*args, **kwargs)
    elif method == Method.gd:
        model = GD(*args, **kwargs)
    else:
        raise Exception("Invalid method")
    return Optimizer(model=model)
