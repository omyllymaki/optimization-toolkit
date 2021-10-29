import logging
from functools import partial

import numpy as np

from src.gauss_newton import GaussNewton

logging.basicConfig(level=logging.INFO)


def f_eval(x, param):
    return param[0] * x + param[1]


def f_err(param, x, y):
    y_estimate = f_eval(x, param)
    return y_estimate - y


def main():
    x = np.arange(1, 100)
    param_true = np.array([1.0, 2.5])
    y = f_eval(x, param_true)
    optimizer = GaussNewton(f_err=partial(f_err, x=x, y=y))
    param, costs, _ = optimizer.run(np.random.randn(2))
    print(f"Param: {param}")
    print(f"Costs: {costs}")


if __name__ == "__main__":
    main()
