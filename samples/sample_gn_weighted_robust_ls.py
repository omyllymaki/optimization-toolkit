import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.gauss_newton import GaussNewton
from src.termination import TerminationCriteria

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

NOISE = 5
OUTLIER_NOISE = 10
OUTLIER_OFFSET = 20
OUTLIER_TREND = 0.8
N_OUTLIERS = 20
PARAMETERS = [0.01, -2, 5]


# Minimize the Lp norm
# e.g. p=1 is L1 norm (least absolute deviation)
def f_weights(errors, eps, p):
    abs_errors = abs(errors)
    abs_errors[abs_errors < eps] = eps
    weights = abs_errors ** (p - 2)
    weights_normalized = len(weights) * weights / np.sum(weights)
    return weights_normalized


def f_eval(x, coeff):
    return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]


def f_err(param, x, y):
    y_estimate = f_eval(x, param)
    return y_estimate - y


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    outlier_indices = np.random.choice(len(x), N_OUTLIERS, replace=False)
    y_outliers = OUTLIER_NOISE * np.random.randn(len(outlier_indices)) + OUTLIER_OFFSET + OUTLIER_TREND * x[
        outlier_indices]
    y_noisy[outlier_indices] = y_noisy[outlier_indices] + y_outliers

    init_guess = np.zeros(3)
    criteria = TerminationCriteria(max_iter=50, cost_diff_threshold=-np.inf)

    optimizer_robust = GaussNewton(f_err=partial(f_err, x=x, y=y_noisy),
                                   f_weights=partial(f_weights, p=1.0, eps=1e-6),
                                   termination=criteria,
                                   step_size_max_iter=0)
    param_robust, costs_robust, _ = optimizer_robust.run(init_guess)
    y_estimate_robust = f_eval(x, param_robust)

    optimizer = GaussNewton(f_err=partial(f_err, x=x, y=y_noisy),
                            f_weights=None,
                            termination=criteria,
                            step_size_max_iter=0)
    param, costs, _ = optimizer.run(init_guess)
    y_estimate = f_eval(x, param)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "b-", label="True", linewidth=2.0)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")
    plt.plot(x, y_estimate, "g-", label="Normal Fit", linewidth=2.0)
    plt.plot(x, y_estimate_robust, "r-", label="Robust Fit", linewidth=2.0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(costs, "g-", label="Normal Fit", linewidth=2.0)
    plt.plot(costs_robust, "r-", label="Robust Fit", linewidth=2.0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
