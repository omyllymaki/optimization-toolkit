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
