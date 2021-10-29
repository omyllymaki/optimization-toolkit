import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.gradient_descent import GradientDescent
from src.termination import TerminationCriteria

logging.basicConfig(level=logging.INFO)

NOISE = 5
PARAMETERS = [0.1, -1, 5]

np.random.seed(42)


def f_eval(x, coeff):
    return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]


def f_step(iter_round, max_iter, max_step):
    r = (iter_round / max_iter) * np.pi
    step_size_ub = np.sin(r) * max_step
    step_size_lb = 0
    return step_size_lb, step_size_ub


# Add constrain to solution by giving penalty for neg param values
def f_cost(param, x, y, neg_penalty=1e8):
    y_eval = f_eval(x, param)
    errors = y_eval - y
    neg_param = param[param < 0]
    mse = np.mean(errors ** 2)
    penalty = neg_penalty * np.sum(neg_param ** 2)
    return mse + penalty


def main():
    x = np.arange(1, 100)

    y = f_eval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    init_guess = - np.random.rand(3)
    max_iter = 5000
    criteria = TerminationCriteria(max_iter=max_iter, cost_diff_threshold=-np.inf, max_iter_without_improvement=1000)
    optimizer = GradientDescent(
        f_cost=partial(f_cost, x=x, y=y_noisy, neg_penalty=1e8),
        termination=criteria,
        f_step=partial(f_step, max_iter=max_iter, max_step=1e-9)
    )
    param, costs, params = optimizer.run(init_guess)
    y_estimate = f_eval(x, param)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "b-", label="Original, noiseless signal", linewidth=1.5)
    plt.plot(x, y_noisy, "k.", label="Noisy signal")
    plt.plot(x, y_estimate, "r-", label="Fit", linewidth=1.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Fit")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(costs, "-")
    plt.yscale("log")
    plt.title("Costs")
    plt.subplot(2, 2, 4)
    plt.plot(params, "-")
    plt.title("Parameters")
    plt.show()


if __name__ == "__main__":
    main()
