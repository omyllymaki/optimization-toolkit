import logging

import numpy as np
from matplotlib import pyplot as plt

from src.optimizer_factory import get_optimizer, Method
from src.termination import TerminationCriteria
from src.utils import mse

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

NOISE = 5
OUTLIER_NOISE = 50
OUTLIER_OFFSET = 20
N_OUTLIERS = 20
PARAMETERS = [-2, 5]


def feval(x, coeff):
    return coeff[0] * x + coeff[1]


def trimmed_cost(errors):
    squared_errors = errors ** 2
    ub = np.percentile(squared_errors, 70)
    i = squared_errors < ub
    return mse(errors[i])


def main():
    x = np.arange(1, 100)

    y = feval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    indices = np.random.choice(len(x), N_OUTLIERS, replace=False)
    y_noisy[indices] = OUTLIER_NOISE * np.random.randn(len(indices)) + OUTLIER_OFFSET

    init_guess = np.zeros(2)
    criteria = TerminationCriteria(max_iter=200)
    step_size = 1e-5
    fstep = lambda _: (0, step_size)
    optimizer_robust = get_optimizer(method=Method.GD,
                                     feval=feval,
                                     fcost=trimmed_cost,
                                     termination_criteria=criteria,
                                     fstep=fstep,
                                     step_size_max_iter=10)
    param_robust, costs_robust, _ = optimizer_robust.fit(x, y_noisy, init_guess)
    y_estimate_robust = feval(x, param_robust)

    optimizer = get_optimizer(method=Method.GD,
                              feval=feval,
                              fcost=mse,
                              termination_criteria=criteria,
                              fstep=fstep,
                              step_size_max_iter=10)
    param, costs, _ = optimizer.fit(x, y_noisy, init_guess)
    y_estimate = feval(x, param)

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
