import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.gradient_descent import GradientDescent
from src.termination import check_n_iter
from src.utils import mse

logging.basicConfig(level=logging.INFO)

NOISE = 5
PARAMETERS = [0.1, -1, 5]

np.random.seed(42)


def f_eval(x, param):
    return param[0] * x ** 2 + param[1] * x + param[2]


def f_step(iter_round, max_iter):
    r = (iter_round / max_iter) * np.pi
    step_size_ub = np.sin(r) * 1e-8  # Max step size will vary from 0 -> 1e-8 -> 0 during iteration
    step_size_lb = 0
    return step_size_lb, step_size_ub


def f_cost(param, x, y):
    y_eval = f_eval(x, param)
    errors = y_eval - y
    return mse(errors)


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    init_guess = np.random.randn(3)
    max_iter = 200
    termination_checks = partial(check_n_iter, threshold=200)

    optimizer = GradientDescent(f_cost=partial(f_cost, x=x, y=y_noisy),
                                termination_checks=termination_checks,
                                f_step=partial(f_step, max_iter=max_iter))
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
