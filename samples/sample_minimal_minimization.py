import logging

import numpy as np

from src.gradient_descent import GradientDescent

logging.basicConfig(level=logging.INFO)


def f_cost(param):
    return (param[0] - 0.5) ** 2 + (param[1] + 0.5) ** 2


def main():
    optimizer = GradientDescent(f_cost=f_cost)
    param, costs, _ = optimizer.run(np.random.randn(2))
    print(f"Param: {param}")
    print(f"Cost: {np.min(costs)}")


if __name__ == "__main__":
    main()
