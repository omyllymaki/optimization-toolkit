import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.gauss_newton import GaussNewton
from src.levenberg_marquardt import LevenbergMarquardt

logging.basicConfig(level=logging.INFO)

np.random.seed(42)


def f_eval(x, coeff):
    return (0.1 * coeff[0] * x + coeff[3] * np.cos(0.5 * coeff[2] * x)) / (coeff[1] + x + np.exp(1e-5 * coeff[4]))


def f_err(param, x, y):
    y_estimate = f_eval(x, param)
    return y_estimate - y


def f_init_guess():
    return np.random.randn(5)


def main():
    x = np.arange(1, 100)
    true_param = np.random.randn(5)
    y = f_eval(x, true_param)
    y_noisy = y + 0.01 * np.random.randn(len(y))
    init_guess = np.random.randn(5)
    gn = GaussNewton(partial(f_err, x=x, y=y_noisy), step_size_max_iter=10)
    lma = LevenbergMarquardt(partial(f_err, x=x, y=y_noisy))

    param_gn, costs_gn, _ = gn.run(init_guess)
    param_lma, costs_lma, _ = lma.run(init_guess)

    y_estimate_gn = f_eval(x, param_gn)
    y_estimate_lma = f_eval(x, param_lma)

    plt.subplot(1, 2, 1)
    plt.plot(x, y_noisy, "b-", label="Noisy input data")
    plt.plot(x, y_estimate_gn, "g-", label="GN")
    plt.plot(x, y_estimate_lma, "r-", label="LMA")
    plt.legend()
    plt.title("Input data & fitted models")
    plt.subplot(1, 2, 2)
    plt.plot(costs_gn, "g-", label="GN")
    plt.plot(costs_lma, "r-", label="LMA")
    plt.legend()
    plt.title("Costs")
    plt.show()


if __name__ == "__main__":
    main()
