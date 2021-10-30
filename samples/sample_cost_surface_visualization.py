import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.gauss_newton import GaussNewton
from src.gradient_descent import GradientDescent
from src.nelder_mead import NelderMead, generate_init_test_points
from src.random_optimization import RandomOptimization
from src.simulated_annealing import SimulatedAnnealing
from src.utils import mse

logging.basicConfig(level=logging.WARNING)

np.random.seed(7)


def f_eval(x, param):
    return (param[0] * x - 2) ** 2 * np.sin(param[1] * x - 4)


def f_err(param, x, y):
    y_estimate = f_eval(x, param)
    return y_estimate - y


def f_cost(param, x, y):
    y_eval = f_eval(x, param)
    errors = y_eval - y
    return mse(errors)


def main():
    param_true = [0.5, 0.1]
    x = np.arange(0, 1, 0.01)
    y = f_eval(x, param_true)

    grid = np.arange(-3, 3, 0.01)

    grid_costs = []
    cost_matrix = np.empty((grid.shape[0], grid.shape[0]))
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            y_eval = f_eval(x, [p1, p2])
            errors = y_eval - y
            cost = mse(errors)

            grid_costs.append(cost)
            cost_matrix[i, j] = cost

    plt.figure(1)
    plt.subplot(1, 2, 1)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.array(grid_costs).reshape(xx.shape)
    plt.pcolormesh(xx, yy, zz ** 0.2)
    plt.colorbar()

    init_guess = np.array([2.5, -2])

    f_step = lambda _: (0, 2.0)
    f_scaling = lambda k: 0.995 ** k * np.ones(2)
    f_update = lambda p, k: p + 0.995 ** k * np.random.randn(2)
    f_temp = lambda k: 0.2 * np.exp(-0.05 * k)
    init_points_nm = generate_init_test_points(init_guess, 0.1)
    optimizers = {
        "GN": (GaussNewton(f_err=partial(f_err, x=x, y=y)), "saddlebrown"),
        "GD": (GradientDescent(f_cost=partial(f_cost, x=x, y=y), f_step=f_step), "darkorange"),
        "RO": (RandomOptimization(f_cost=partial(f_cost, x=x, y=y), f_scaling=f_scaling), "red"),
        "SA": (SimulatedAnnealing(f_cost=partial(f_cost, x=x, y=y), f_update=f_update, f_temp=f_temp), "cyan"),
        "NM": (NelderMead(f_cost=partial(f_cost, x=x, y=y), init_test_points=init_points_nm), "green")
    }

    for name, (optimizer, color) in optimizers.items():
        print(f"Running {name}")
        param, costs, params = optimizer.run(init_guess.copy())
        plt.subplot(1, 2, 1)
        plt.plot(params[:, 0], params[:, 1], "-", label=name, color=color)
        plt.ylabel("Param 2")
        plt.xlabel("Param 1")
        plt.title("Cost^(1/5)")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(costs, label=name, color=color)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Cost")
        plt.xlabel("Iter")
        plt.title("Cost vs iter")
        plt.legend()
        plt.grid()

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(init_guess[0], init_guess[1], "ko")
    plt.plot(param_true[0], param_true[1], "ko")
    plt.show()


if __name__ == "__main__":
    main()
