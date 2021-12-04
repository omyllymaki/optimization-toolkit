import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.nelder_mead import NelderMead
from src.termination import check_n_iter, check_n_iter_without_improvement
from src.loss import eq_constraint_penalty, ieq_constraint_penalty

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
def f_cost(x):
    h = np.array([1, 1, 1, 2]).reshape(2, 2)
    g = np.array([1, 2])
    r = 2
    return 0.5 * x.T @ h @ x + g.T @ x + r


# constrained function to minimize
def f_cost_constrained(x):
    f = f_cost(x)
    p1 = eq_constraint_penalty(eq_constraint, x, EQ_CONSTRAINT_PENALTY)
    p2 = ieq_constraint_penalty(ieq_constraint, x, IEQ_CONSTRAINT_PENALTY)
    return f + p1 + p2


def main():
    g = np.arange(-10, 10, 0.1)

    grid_costs = []
    for i, x1 in enumerate(g):
        for j, x2 in enumerate(g):
            cost = f_cost(np.array([x1, x2]))
            grid_costs.append(cost)

    termination = (
        partial(check_n_iter, threshold=100),
        partial(check_n_iter_without_improvement, threshold=10),
    )

    init_guess = np.array([5, 5]).astype(float)
    optimizer = NelderMead(f_cost=f_cost_constrained, termination_checks=termination)
    output = optimizer.run(init_guess)

    plt.subplot(1, 2, 1)
    xx, yy = np.meshgrid(g, g)
    zz = np.array(grid_costs).reshape(xx.shape)
    plt.pcolormesh(xx, yy, zz ** 0.2)
    plt.colorbar()
    plt.plot(g, g - 8, "r--", label="Equality constraint")
    plt.plot(g, -2 * g - 4, "g--", label="Inequality constraint")
    plt.plot(output.xs[:, 0], output.xs[:, 1], "k-")
    plt.plot(output.x[0], output.x[1], "mo", markersize=5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(output.costs)
    plt.yscale("log")

    plt.show()


if __name__ == "__main__":
    main()
