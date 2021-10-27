import logging

import numpy as np

from src.gauss_newton import GaussNewton

logging.basicConfig(level=logging.INFO)


def f_eval(x, param):
    return param[0] * x + param[1]


def main():
    x = np.arange(1, 100)
    param_true = np.array([1.0, 2.5])
    y = f_eval(x, param_true)
    param, costs, _ = GaussNewton(f_eval=f_eval).run(x, y, np.random.randn(2))
    print(f"Param: {param}")
    print(f"Costs: {costs}")


if __name__ == "__main__":
    main()
