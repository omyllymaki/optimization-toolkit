import logging
import warnings
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.levenberg_marquardt import LevenbergMarquardt
from src.termination import check_n_iter
from src.utils import generalized_robust_loss

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
np.random.seed(42)

NOISE = 5
OUTLIER_NOISE = 10
OUTLIER_OFFSET = 20
OUTLIER_TREND = 0.8
N_OUTLIERS = 20
PARAMETERS = [0.01, -2, 5]


def f_eval(x, coeff):
    return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]


def f_err(param, x, y, loss_alpha, loss_scale=1.0):
    y_estimate = f_eval(x, param)
    diff = y_estimate - y
    return generalized_robust_loss(diff, loss_alpha, loss_scale)


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    outlier_indices = np.random.choice(len(x), N_OUTLIERS, replace=False)
    y_outliers = OUTLIER_NOISE * np.random.randn(len(outlier_indices)) + OUTLIER_OFFSET + OUTLIER_TREND * x[
        outlier_indices]
    y_noisy[outlier_indices] = y_noisy[outlier_indices] + y_outliers

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(x, y, "k-", label="True", linewidth=2.0)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(x, y, "k-", label="True", linewidth=2.0)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")

    init_guess = np.zeros(3)
    termination_checks = partial(check_n_iter, threshold=200)

    alphas = [2, 1.5, 1, 0, -1]
    for alpha in alphas:
        fe = partial(f_err, x=x, y=y_noisy, loss_scale=1.0, loss_alpha=alpha)
        optimizer = LevenbergMarquardt(f_err=fe, termination_checks=termination_checks)
        output = optimizer.run(init_guess)
        y_estimate = f_eval(x, output.x)

        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(x, y_estimate, label=f"alpha {alpha}", linewidth=2.0)
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(output.costs, label=f"alpha {alpha}", linewidth=2.0)
        plt.legend()
        plt.yscale("log")
        plt.subplot(2, 2, 4)
        plt.plot(y_estimate - y, label=f"alpha {alpha}", linewidth=2.0)
        plt.legend()

    scales = [1e3, 100, 10, 1, 0.1, 0.01]
    for scale in scales:
        fe = partial(f_err, x=x, y=y_noisy, loss_scale=scale, loss_alpha=1)
        optimizer = LevenbergMarquardt(f_err=fe, termination_checks=termination_checks)
        output = optimizer.run(init_guess)
        y_estimate = f_eval(x, output.x)

        plt.figure(2)
        plt.subplot(1, 2, 1)
        plt.plot(x, y_estimate, label=f"scale {scale}", linewidth=2.0)
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(output.costs, label=f"scale {scale}", linewidth=2.0)
        plt.legend()
        plt.yscale("log")
        plt.subplot(2, 2, 4)
        plt.plot(y_estimate - y, label=f"scale {scale}", linewidth=2.0)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
