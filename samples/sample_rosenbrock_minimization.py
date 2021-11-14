import logging

import numpy as np
from matplotlib import pyplot as plt

from src.local_optimization.gauss_newton import GaussNewton
from src.local_optimization.gradient_descent import GradientDescent
from src.local_optimization.levenberg_marquardt import LevenbergMarquardt
from src.local_optimization.nelder_mead import NelderMead
from src.local_optimization.random_optimization import RandomOptimization
from src.local_optimization.simulated_annealing import SimulatedAnnealing

logging.basicConfig(level=logging.INFO)

np.random.seed(42)


def f_update(param, k):
    scaling_factors = 0.995 ** k * np.array([0.2, 0.2])
    return param + scaling_factors * np.random.randn(param.shape[0])


def f_cost(param, a=1, b=10):
    return (a - param[0]) ** 2 + b * (param[1] - param[0] ** 2) ** 2


def f_step(iter_round, max_iter=10000):
    r = (iter_round / max_iter) * np.pi
    step_size_ub = np.sin(r) * 2e-2  # Max step size will vary from 0 -> 1e-4
    step_size_lb = 0
    return 0, 2e-2


# This is needed for Gauss-newton
# Here we define f_err so that f_cost = mse(f_err) = sum(f_err^2)
# In practice, f_cost = 0.5*(e1^2 + e2^2) = (a - param[0])^2 + b*(param[1] - param[0]^2)^2
def f_err(param, a=1, b=100):
    e1 = a - param[0]
    e2 = np.sqrt(b) * (param[1] - param[0] ** 2)
    return np.sqrt(2) * np.array([e1, e2])


def main():
    true_minimum = np.array([1, 1])

    grid = np.arange(-5, 5, 0.01)

    grid_costs = []
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            cost = f_cost(np.array([p1, p2]))
            grid_costs.append(cost)

    init_guess = np.array([-2.0, 2.0])

    f_step = lambda _: (0, 2e-2)
    f_scaling = lambda k: 0.995 ** k * np.ones(2)
    f_update = lambda p, k: p + 0.999 ** k * np.random.randn(2)
    f_temp = lambda k: 1.0 * np.exp(-0.01 * k)
    optimizers = {
        "GD": (GradientDescent(f_cost=f_cost, f_step=f_step), "darkorange"),
        "GN": (GaussNewton(f_err=f_err, step_size_max_iter=0), "darkblue"),
        "LMA": (LevenbergMarquardt(f_err=f_err), "olive"),
        "RO": (RandomOptimization(f_cost=f_cost, f_scaling=f_scaling), "red"),
        "SA": (SimulatedAnnealing(f_cost=f_cost, f_update=f_update, f_temp=f_temp), "cyan"),
        "NM": (NelderMead(f_cost=f_cost), "darkviolet")
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
        plt.xscale("log")
        plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(init_guess[0], init_guess[1], "ko")
    plt.plot(true_minimum[0], true_minimum[1], "ko")

    plt.show()


if __name__ == "__main__":
    main()
