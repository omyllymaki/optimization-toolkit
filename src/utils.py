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


def eq_constraint_penalty(f_constraint: Callable, x: np.ndarray, penalty_parameter=1e6):
    """
    Calculate soft equality constraint penalty.

    @param f_constraint: constraint that should be satisfied; constrain(x) = 0.
    @param x: variables
    @param penalty_parameter: penalty parameter.
    @return: penalty.
    """
    return penalty_parameter * f_constraint(x) ** 2


def ieq_constraint_penalty(f_constraint: Callable, x: np.ndarray, penalty_parameter=1e6):
    """
    Calculate soft inequality constraint penalty.

    @param f_constraint: inequality constraint that should be satisfied; f_constraint(x) <= 0.
    @param x: variables
    @param penalty_parameter: penalty parameter.
    @return: penalty.
    """
    return penalty_parameter * max(0, f_constraint(x)) ** 2
