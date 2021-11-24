import logging

import numpy as np

from src.local_optimization.gradient_descent import GradientDescent

logging.basicConfig(level=logging.INFO)


def f_cost(x):
    return (x[0] - 0.5) ** 2 + (x[1] + 0.5) ** 2


def main():
    optimizer = GradientDescent(f_cost=f_cost)
    output = optimizer.run(np.random.randn(2))
    print(f"Param: {output.x}")
    print(f"Cost: {output.min_cost}")


if __name__ == "__main__":
    main()
