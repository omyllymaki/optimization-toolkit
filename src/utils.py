from typing import Callable

import numpy as np
from numpy.linalg import pinv


def gradient(x0: np.ndarray,
             f: Callable,
             step: float = 1e-9) -> np.ndarray:
    """
    Calculate gradient matrix numerically.
    """
    y0 = f(x0)
    output = []
    for i in range(len(x0)):
        xi = x0.copy()
        xi[i] += step
        yi = f(xi)
        derivative = (yi - y0) / step
        output.append(derivative)
    return np.array(output).T


def pseudoinverse(x: np.ndarray) -> np.ndarray:
    """
    Moore-Penrose inverse.
    """
    return pinv(x.T @ x) @ x.T


def diff(y_fit, y):
    return y_fit - y


def rmse(errors):
    return np.sqrt(mse(errors))


def mse(errors):
    return np.mean(errors ** 2)


def add_eq_constraint(func: Callable, constrain: Callable, penalty_parameter=1e6):
    """
    Add soft equality constrain to function func.

    @param func: Original function.
    @param constrain: constrain that should be satisfied; constrain(x) = 0.
    @param penalty_parameter: penalty parameter.
    @return: Original function that is augmented with soft equality constrain.
    """
    return lambda x: func(x) + penalty_parameter * constrain(x) ** 2


def add_ieq_constraint(func: Callable, constrain: Callable, penalty_parameter=1e6):
    """
    Add soft inequality constrain to function func.

    @param func: Original function.
    @param constrain: constrain that should be satisfied; constrain(x) <= 0.
    @param penalty_parameter: penalty parameter.
    @return: Original function that is augmented with soft inequality constrain.
    """
    return lambda x: func(x) + penalty_parameter * max(0, constrain(x)) ** 2
