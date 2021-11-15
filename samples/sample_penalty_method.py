import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.nelder_mead import NelderMead
from src.termination import check_n_iter, check_n_iter_without_improvement
from src.utils import add_eq_constraint, add_ieq_constraint

logging.basicConfig(level=logging.INFO)

EQ_CONSTRAINT_PENALTY = 50.0
IEQ_CONSTRAINT_PENALTY = 50.0


# equality constraint: x2 - x1 + 8 = 0
def eq_constraint(x):
    x1, x2 = x
    return x2 - x1 + 8


# inequality constraint: 2 * x1 + x2 + 4 <= 0
def ieq_constraint(x):
    x1, x2 = x
    return 2 * x1 + x2 + 4


# unconstrained function to minimize
def f(x):
    h = np.array([1, 1, 1, 2]).reshape(2, 2)
    g = np.array([1, 2])
    r = 2
    return 0.5 * x.T @ h @ x + g.T @ x + r


def main():
    g = np.arange(-10, 10, 0.1)

    grid_costs = []
    for i, x1 in enumerate(g):
        for j, x2 in enumerate(g):
            cost = f(np.array([x1, x2]))
            grid_costs.append(cost)

    termination = (
        partial(check_n_iter, threshold=100),
        partial(check_n_iter_without_improvement, threshold=10),
    )

    init_guess = np.array([5, 5]).astype(float)
    f_cost = f
    f_cost = add_eq_constraint(f_cost, eq_constraint, penalty_parameter=EQ_CONSTRAINT_PENALTY)
    f_cost = add_ieq_constraint(f_cost, ieq_constraint, penalty_parameter=IEQ_CONSTRAINT_PENALTY)
    optimizer = NelderMead(f_cost=f_cost, termination_checks=termination)
    x, costs, xs = optimizer.run(init_guess)

    plt.subplot(1, 2, 1)
    xx, yy = np.meshgrid(g, g)
    zz = np.array(grid_costs).reshape(xx.shape)
    plt.pcolormesh(xx, yy, zz ** 0.2)
    plt.colorbar()
    plt.plot(g, g - 8, "r--", label="Equality constraint")
    plt.plot(g, -2 * g - 4, "g--", label="Inequality constraint")
    plt.plot(xs[:, 0], xs[:, 1], "k-")
    plt.plot(x[0], x[1], "mo", markersize=5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(costs)
    plt.yscale("log")

    plt.show()


if __name__ == "__main__":
    main()
