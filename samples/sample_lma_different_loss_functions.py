import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.levenberg_marquardt import LevenbergMarquardt
from src.termination import check_n_iter

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

NOISE = 5
OUTLIER_NOISE = 10
OUTLIER_OFFSET = 20
OUTLIER_TREND = 0.8
N_OUTLIERS = 20
PARAMETERS = [0.01, -2, 5]


def f_eval(x, coeff):
    return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]


def linear(errors):
    return errors


def soft_l1(errors, eps=0):
    abs_errors = abs(errors)
    abs_errors[abs_errors < eps] = eps
    rho = 2 * ((1 + abs_errors) ** 0.5 - 1)
    return rho


def cauchy(errors, eps=0):
    abs_errors = abs(errors)
    abs_errors[abs_errors < eps] = eps
    return np.log(1 + abs_errors)


def f_err(param, x, y, f_loss=soft_l1):
    y_estimate = f_eval(x, param)
    diff = y_estimate - y
    return f_loss(diff)


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    outlier_indices = np.random.choice(len(x), N_OUTLIERS, replace=False)
    y_outliers = OUTLIER_NOISE * np.random.randn(len(outlier_indices)) + OUTLIER_OFFSET + OUTLIER_TREND * x[
        outlier_indices]
    y_noisy[outlier_indices] = y_noisy[outlier_indices] + y_outliers

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "r-", label="True", linewidth=2.0)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")

    init_guess = np.zeros(3)

    termination_checks = partial(check_n_iter, threshold=50)

    loss_functions = [linear, soft_l1, cauchy]

    for loss_function in loss_functions:
        fe = partial(f_err, x=x, y=y_noisy, f_loss=loss_function)
        optimizer = LevenbergMarquardt(f_err=fe, termination_checks=termination_checks)
        output = optimizer.run(init_guess)
        y_estimate = f_eval(x, output.x)

        plt.subplot(1, 2, 1)
        plt.plot(x, y_estimate, label=loss_function.__name__, linewidth=2.0)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(output.costs, label=loss_function.__name__, linewidth=2.0)
        plt.legend()
        plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
