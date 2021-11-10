import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from src.local_optimization.gauss_newton import GaussNewton

logging.basicConfig(level=logging.INFO)

NOISE = 3
PARAMETERS = [-0.001, 0.1, 0.1, 2, 15]


def f_eval(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * np.sin(x)


def f_err(param, x, y):
    y_estimate = f_eval(x, param)
    return y_estimate - y


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    init_guess = 1000000 * np.random.random(len(PARAMETERS))
    optimizer = GaussNewton(f_err=partial(f_err, x=x, y=y_noisy))
    param, costs, _ = optimizer.run(init_guess)
    y_estimate = f_eval(x, param)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, label="Original, noiseless signal", linewidth=1.5)
    plt.plot(x, y_noisy, label="Noisy signal", linewidth=1.5)
    plt.plot(x, y_estimate, label="Fit", linewidth=1.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(costs, "-o")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
