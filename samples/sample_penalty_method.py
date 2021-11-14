import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.nelder_mead import NelderMead
from src.termination import check_n_iter, check_n_iter_without_improvement

logging.basicConfig(level=logging.INFO)

EQ_CONSTRAINT_PENALTY = 50.0
IEQ_CONSTRAINT_PENALTY = 50.0


# equality constraint: x2 - x1 + 8 = 0
def eq_constraint(param):
    x1, x2 = param
    return x2 - x1 + 8


# inequality constraint: 2 * x1 + x2 + 4 <= 0
def ieq_constraint(param):
    x1, x2 = param
    return 2 * x1 + x2 + 4


# unconstrained function to minimize
def f(param):
    h = np.array([1, 1, 1, 2]).reshape(2, 2)
    g = np.array([1, 2])
    r = 2
    return 0.5 * param.T @ h @ param + g.T @ param + r


# constrained function to minimize
def f_constrained(param, a1, a2):
    p1 = a1 * eq_constraint(param) ** 2
    p2 = a2 * max(0.0, ieq_constraint(param)) ** 2
    return f(param) + p1 + p2


def main():
    x = np.arange(-10, 10, 0.1)

    grid_costs = []
    for i, x1 in enumerate(x):
        for j, x2 in enumerate(x):
            cost = f(np.array([x1, x2]))
            grid_costs.append(cost)

    termination = (
        partial(check_n_iter, threshold=100),
        partial(check_n_iter_without_improvement, threshold=10),
    )

    init_guess = np.array([5, 5]).astype(float)
    f_cost = partial(f_constrained, a1=EQ_CONSTRAINT_PENALTY, a2=IEQ_CONSTRAINT_PENALTY)
    optimizer = NelderMead(f_cost=f_cost, termination_checks=termination)
    param, costs, params = optimizer.run(init_guess)

    plt.subplot(1, 2, 1)
    xx, yy = np.meshgrid(x, x)
    zz = np.array(grid_costs).reshape(xx.shape)
    plt.pcolormesh(xx, yy, zz ** 0.2)
    plt.colorbar()
    plt.plot(x, x - 8, "r--")
    plt.plot(x, -2 * x - 4, "r--")
    plt.plot(params[:, 0], params[:, 1], "k-")
    plt.plot(param[0], param[1], "mo", markersize=5)

    plt.subplot(1, 2, 2)
    plt.plot(costs)
    plt.yscale("log")

    plt.show()


if __name__ == "__main__":
    main()
