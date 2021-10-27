import logging

import numpy as np
from matplotlib import pyplot as plt

from src.gauss_newton import GaussNewton

logging.basicConfig(level=logging.INFO)

np.random.seed(21)


def residual_func(y_estimate, y):
    return y_estimate - y


def f1(x, coeff):
    return coeff[0] * x + coeff[1] * np.sin(x)


def f2(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2]


def f3(x, coeff):
    return coeff[0] * np.exp(1e-2 * coeff[1] * 1 / x) - coeff[2] * x ** 2


def f4(x, coeff):
    return 0.1 * coeff[0] * x / (coeff[1] + x)


def f5(x, coeff):
    return (0.1 * coeff[0] * x + coeff[3] * np.cos(0.5 * coeff[2] * x)) / (coeff[1] + x + np.exp(1e-5 * coeff[4]))


fset = (
    (f1, 2),
    (f2, 3),
    (f3, 3),
    (f4, 2),
    (f5, 5),
)

x = np.arange(1, 100)

for f, dim in fset:

    plt.figure()
    for k in range(16):
        print(f"Test run {k}")
        true_param = np.random.randn(dim)
        y = f(x, true_param)
        optimizer = GaussNewton(f_eval=f, step_size_max_iter=20)
        for i in range(50):
            init_guess = np.random.rand(dim)
            param, costs, _ = optimizer.fit(x, y, init_guess)
            y_estimate = f(x, param)
            residual = y_estimate - y
            rmse = np.sqrt(np.sum(residual ** 2))

            if rmse < 1e-3:
                break

        plt.subplot(4, 4, k + 1)
        plt.plot(x, y, "b-")
        plt.plot(x, y_estimate, "r-")
        plt.title(f"{rmse:0.3f}")

plt.show()
