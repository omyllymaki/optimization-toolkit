import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.gradient_descent import GradientDescent
from src.termination import check_n_iter
from src.loss import mse

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

NOISE = 5
OUTLIER_NOISE = 50
OUTLIER_OFFSET = 20
N_OUTLIERS = 20
PARAMETERS = [-2, 5]


def f_eval(x, coeff):
    return coeff[0] * x + coeff[1]


def trimmed_cost(param, x, y, threshold):
    y_eval = f_eval(x, param)
    errors = y_eval - y
    squared_errors = errors ** 2
    ub = np.percentile(squared_errors, threshold)
    i = squared_errors < ub
    return mse(errors[i])


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    indices = np.random.choice(len(x), N_OUTLIERS, replace=False)
    y_noisy[indices] = OUTLIER_NOISE * np.random.randn(len(indices)) + OUTLIER_OFFSET

    init_guess = np.zeros(2)
    termination_checks = partial(check_n_iter, threshold=200)
    step_size = 1e-5
    f_step = lambda _: (0, step_size)
    optimizer_robust = GradientDescent(f_cost=partial(trimmed_cost, x=x, y=y_noisy, threshold=70),
                                       termination_checks=termination_checks,
                                       f_step=f_step,
                                       step_size_max_iter=10)
    output_robust = optimizer_robust.run(init_guess)
    y_estimate_robust = f_eval(x, output_robust.x)

    optimizer = GradientDescent(f_cost=partial(trimmed_cost, x=x, y=y_noisy, threshold=100),
                                termination_checks=termination_checks,
                                f_step=f_step,
                                step_size_max_iter=10)
    output = optimizer.run(init_guess)
    y_estimate = f_eval(x, output.x)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "b-", label="Original, noiseless signal", linewidth=1.5)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")
    plt.plot(x, y_estimate_robust, "r-", label="Robust Fit", linewidth=1.5)
    plt.plot(x, y_estimate, "g-", label="Normal fit", linewidth=1.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(output_robust.costs, "r-", label="Robust fit")
    plt.plot(output.costs, "g-", label="Normal fit")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
