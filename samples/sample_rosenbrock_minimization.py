import logging

import numpy as np
from matplotlib import pyplot as plt

from src.gradient_descent import GradientDescent
from src.random_optimization import RandomOptimization
from src.simulated_annealing import SimulatedAnnealing
from src.termination import TerminationCriteria

logging.basicConfig(level=logging.INFO)

np.random.seed(42)


def f_update(param, k):
    scaling_factors = 0.995 ** k * np.array([0.2, 0.2])
    return param + scaling_factors * np.random.randn(param.shape[0])


def f_cost(param, a=1, b=100):
    return (a - param[0]) ** 2 + b * (param[1] - param[0] ** 2) ** 2


def f_step(iter_round, max_iter=10000):
    r = (iter_round / max_iter) * np.pi / 2
    step_size_ub = np.sin(r) * 5e-2  # Max step size will vary from 0 -> 1e-4
    step_size_lb = 0
    return step_size_lb, step_size_ub


def main():
    true_minimum = np.array([1, 1])

    grid = np.arange(-3, 3, 0.01)

    grid_costs = []
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            cost = f_cost(np.array([p1, p2]))
            grid_costs.append(cost)

    init_guess = np.array([0, -2])
    criteria = TerminationCriteria(max_iter=10000, cost_diff_threshold=-np.inf, max_iter_without_improvement=5000)

    f_scaling = lambda k: 0.995 ** k * np.ones(2)
    f_update = lambda p, k: p + 0.999 ** k * np.random.randn(2)
    f_temp = lambda k: 1.0 * np.exp(-0.01 * k)
    optimizers = {
        "GD": (GradientDescent(f_cost=f_cost, f_step=f_step, termination=criteria), "darkorange"),
        "RO": (RandomOptimization(f_cost=f_cost, f_scaling=f_scaling, termination=criteria), "red"),
        "SA": (SimulatedAnnealing(f_cost=f_cost, f_update=f_update, f_temp=f_temp, termination=criteria), "cyan")
    }

    plt.subplot(1, 2, 1)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.array(grid_costs).reshape(xx.shape)
    plt.pcolormesh(xx, yy, zz ** 0.2)
    plt.colorbar()

    for name, (optimizer, color) in optimizers.items():
        param, costs, params = optimizer.run(init_guess)
        plt.subplot(1, 2, 1)
        plt.plot(params[:, 0], params[:, 1], label=name, color=color)
        plt.plot(params[-1, 0], params[-1, 1], color=color, marker="x")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(costs, "-", label=name, color=color)
        plt.yscale("log")
        plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(init_guess[0], init_guess[1], "ko")
    plt.plot(true_minimum[0], true_minimum[1], "ko")

    plt.show()


if __name__ == "__main__":
    main()
