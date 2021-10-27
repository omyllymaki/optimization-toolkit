import logging

import numpy as np
from matplotlib import pyplot as plt

from src.optimizer_factory import get_optimizer, Method
from src.termination import TerminationCriteria

logging.basicConfig(level=logging.INFO)

NOISE = 5
PARAMETERS = [0.1, -1, 5]


def f_eval(x, coeff):
    return coeff[0] * x**2 + coeff[1]*x + coeff[2]


def f_scaling(k):
    return 0.99 ** k * np.array([0.1, 1, 1])


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    init_guess = np.zeros(3)
    criteria = TerminationCriteria(max_iter=1000, cost_diff_threshold=-np.inf)
    optimizer = get_optimizer(method=Method.RO, f_eval=f_eval, f_scaling=f_scaling, termination_criteria=criteria)
    param, costs, _ = optimizer.fit(x, y_noisy, init_guess)
    y_estimate = f_eval(x, param)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "b-", label="Original, noiseless signal", linewidth=1.5)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")
    plt.plot(x, y_estimate, "r-", label="Fit", linewidth=1.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(costs, "-o")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
