import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.gauss_newton import GaussNewton
from src.termination import check_n_iter

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

    termination_checks = partial(check_n_iter, threshold=50)

    optimizer_robust = GaussNewton(f_err=partial(f_err, x=x, y=y_noisy),
                                   f_weights=partial(f_weights, p=1.0, eps=1e-6),
                                   step_size_max_iter=0,
                                   termination_checks=termination_checks)
    output_robust = optimizer_robust.run(init_guess)
    y_estimate_robust = f_eval(x, output_robust.x)

    optimizer = GaussNewton(f_err=partial(f_err, x=x, y=y_noisy),
                            f_weights=None,
                            step_size_max_iter=0,
                            termination_checks=termination_checks)
    output = optimizer.run(init_guess)
    y_estimate = f_eval(x, output.x)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "b-", label="True", linewidth=2.0)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")
    plt.plot(x, y_estimate, "g-", label="Normal Fit", linewidth=2.0)
    plt.plot(x, y_estimate_robust, "r-", label="Robust Fit", linewidth=2.0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(output.costs, "g-", label="Normal Fit", linewidth=2.0)
    plt.plot(output_robust.costs, "r-", label="Robust Fit", linewidth=2.0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
