import logging
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src.optimizer_factory import get_optimizer, Method
from src.termination import TerminationCriteria
from src.utils import mse

logging.basicConfig(level=logging.INFO)

NOISE = 5
PARAMETERS = [0.1, -1, 5]

np.random.seed(42)


def feval(x, coeff):
    return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]


def fstep(iter_round, max_iter, max_step):
    r = (iter_round / max_iter) * np.pi
    step_size_ub = np.sin(r) * max_step
    step_size_lb = 0
    return step_size_lb, step_size_ub


# Add constrain to solution by giving penalty for neg param values
def fcost(errors, param, neg_penalty=1e8):
    neg_param = param[param < 0]
    mse = np.mean(errors ** 2)
    penalty = neg_penalty * np.sum(neg_param ** 2)
    return mse + penalty


def main():
    x = np.arange(1, 100)

    y = feval(x, PARAMETERS)
    y_noisy = y + NOISE * np.random.randn(len(x))

    init_guess = - np.random.rand(3)
    max_iter = 5000
    criteria = TerminationCriteria(max_iter=max_iter, cost_diff_threshold=-np.inf, max_iter_without_improvement=1000)
    optimizer = get_optimizer(method=Method.GD,
                              feval=feval,
                              fcost=partial(fcost, neg_penalty=1e8),
                              termination_criteria=criteria,
                              fstep=partial(fstep, max_iter=max_iter, max_step=1e-9))
    param, costs, params = optimizer.fit(x, y_noisy, init_guess)
    y_estimate = feval(x, param)

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
