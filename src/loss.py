from typing import Callable

import numpy as np


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


def generalized_robust_kernel(errors: np.ndarray,
                              alpha: float,
                              scale: float) -> np.ndarray:
    """
    Generalized robust losses, based on kernel specification.

    @param errors: Errors.
    @param alpha:  Shape parameter.
    @param scale: Size of quadratic loss region around zero errors.
    @return: Weighted errors.
    """
    if alpha >= 2:
        return errors
    z = errors ** 2 / scale ** 2
    if alpha == 0:
        out = np.log(z / 2 + 1)
    else:
        t1 = np.abs(alpha - 2) / alpha
        t2 = (z / abs(alpha - 2) + 1) ** (alpha / 2) - 1
        out = t1 * t2
    out = np.sqrt(out)
    i_neg = errors < 0
    out[i_neg] = -1 * out[i_neg]
    return out