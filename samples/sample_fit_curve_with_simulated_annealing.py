import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.simulated_annealing import SimulatedAnnealing
from src.termination import TerminationCriteria
from src.utils import mse

logging.basicConfig(level=logging.INFO)

NOISE = 5
PARAMETERS = [0.1, -1, 5]


def f_eval(x, coeff):
    return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]


def f_update(param, k):
    scaling_factors = 0.99 ** k * np.array([0.1, 1, 1])
    return param + scaling_factors * np.random.randn(param.shape[0])


def f_cost(param, x, y):
    y_eval = f_eval(x, param)
    errors = y_eval - y
    return mse(errors)


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    init_guess = np.zeros(3)
    criteria = TerminationCriteria(max_iter=1000, cost_diff_threshold=-np.inf, max_iter_without_improvement=200)
    optimizer = SimulatedAnnealing(f_cost=partial(f_cost, x=x, y=y_noisy),
                                   f_update=f_update,
                                   termination=criteria)
    param, costs, _ = optimizer.run(init_guess)
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
