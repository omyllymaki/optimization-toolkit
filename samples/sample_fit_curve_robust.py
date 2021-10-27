import logging

import numpy as np
from matplotlib import pyplot as plt

from src.gradient_descent import GradientDescent
from src.termination import TerminationCriteria
from src.utils import mse

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

NOISE = 5
OUTLIER_NOISE = 50
OUTLIER_OFFSET = 20
N_OUTLIERS = 20
PARAMETERS = [-2, 5]


def f_eval(x, coeff):
    return coeff[0] * x + coeff[1]


def trimmed_cost(errors, param):
    squared_errors = errors ** 2
    ub = np.percentile(squared_errors, 70)
    i = squared_errors < ub
    return mse(errors[i], param)


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    indices = np.random.choice(len(x), N_OUTLIERS, replace=False)
    y_noisy[indices] = OUTLIER_NOISE * np.random.randn(len(indices)) + OUTLIER_OFFSET

    init_guess = np.zeros(2)
    criteria = TerminationCriteria(max_iter=200)
    step_size = 1e-5
    f_step = lambda _: (0, step_size)
    optimizer_robust = GradientDescent(f_eval=f_eval,
                                       f_cost=trimmed_cost,
                                       termination=criteria,
                                       f_step=f_step,
                                       step_size_max_iter=10)
    param_robust, costs_robust, _ = optimizer_robust.run(x, y_noisy, init_guess)
    y_estimate_robust = f_eval(x, param_robust)

    optimizer = GradientDescent(f_eval=f_eval,
                                f_cost=mse,
                                termination=criteria,
                                f_step=f_step,
                                step_size_max_iter=10)
    param, costs, _ = optimizer.run(x, y_noisy, init_guess)
    y_estimate = f_eval(x, param)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "b-", label="Original, noiseless signal", linewidth=1.5)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")
    plt.plot(x, y_estimate_robust, "r-", label="Robust Fit", linewidth=1.5)
    plt.plot(x, y_estimate, "g-", label="Normal fit", linewidth=1.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(costs_robust, "r-", label="Robust fit")
    plt.plot(costs, "g-", label="Normal fit")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
