import logging
from functools import partial

import numpy as np

from src.local_optimization.gradient_descent import GradientDescent

logging.basicConfig(level=logging.INFO)

np.random.seed(21)


def f_eval(x, param):
    return param[0] * x + param[1]


def f_cost(param, x, y):
    y_estimate = f_eval(x, param)
    errors = y_estimate - y
    return np.mean(errors ** 2)


def main():
    x = np.arange(1, 100)
    param_true = np.array([1.0, 2.5])
    y = f_eval(x, param_true)
    f_step = lambda k: (0, 1e-3)
    optimizer = GradientDescent(f_cost=partial(f_cost, x=x, y=y), f_step=f_step)
    param, costs, _ = optimizer.run(np.random.randn(2))
    print(f"Param: {param}")
    print(f"Cost: {np.min(costs)}")


if __name__ == "__main__":
    main()
