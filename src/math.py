from typing import Callable

import numpy as np


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
