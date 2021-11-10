import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.global_optimization.multi_start_optimizer import MultiStartOptimizer
from src.local_optimization.gauss_newton import GaussNewton

logging.basicConfig(level=logging.INFO)

np.random.seed(42)


def f_eval(x, coeff):
    return (0.1 * coeff[0] * x + coeff[3] * np.cos(0.5 * coeff[2] * x)) / (coeff[1] + x + np.exp(1e-5 * coeff[4]))


def f_err(param, x, y):
    y_estimate = f_eval(x, param)
    return y_estimate - y


def f_init_guess(params, costs):
    return np.random.randn(5)


def main():
    x = np.arange(1, 100)
    true_param = np.random.randn(5)
    y = f_eval(x, true_param)
    y_noisy = y + 0.01 * np.random.randn(len(y))

    local_optimizer = GaussNewton(partial(f_err, x=x, y=y_noisy), step_size_max_iter=5)
    global_optimizer = MultiStartOptimizer(optimizer=local_optimizer, f_init_guess=f_init_guess)

    best_param, costs, all_params = global_optimizer.run()
    i_min = np.argmin(costs)
    i_max = np.argmax(costs)

    y_estimate = f_eval(x, all_params[0])
    y_estimate_best = f_eval(x, all_params[i_min])
    y_estimate_worst = f_eval(x, all_params[i_max])

    plt.subplot(1, 2, 1)
    plt.plot(x, y_noisy, "b-", label="Noisy input data")
    plt.plot(x, y_estimate, "g-", label="Estimate, first run")
    plt.plot(x, y_estimate_best, "r-", label="Estimate, best run")
    plt.plot(x, y_estimate_worst, "m-", label="Estimate, worst run")
    plt.legend()
    plt.title("Input data & fitted models")
    plt.subplot(1, 2, 2)
    plt.plot(costs)
    plt.title("Final cost for different init guesses")
    plt.show()


if __name__ == "__main__":
    main()
