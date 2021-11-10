import logging
from functools import partial
from typing import Callable, Tuple

import numpy as np

from src.optimizer import Optimizer
from src.termination import check_n_iter, check_n_iter_without_improvement, check_absolute_cost

TERMINATION_CHECKS = (
    partial(check_n_iter, threshold=1000),
    partial(check_n_iter_without_improvement, threshold=200),
    partial(check_absolute_cost, threshold=1e-6)
)

logger = logging.getLogger(__name__)


def generate_init_test_points(param, scale=1):
    """
    Generate initial test points around param.
    """
    n_dim = len(param)
    points = param.copy()
    for k in range(n_dim):
        v = np.random.randn(len(points))
        v = v / np.linalg.norm(v)
        point = param + scale * v
        points = np.vstack((points, point))
    return points


def get_factors(n_dim):
    """
    Helper function to get factors for Nelder-Mead optimizer, based on the dimension of the problem.
    """
    reflection_factor = 1.0
    expansion_factor = 1 + 2 / n_dim
    contraction_factor = 0.75 - 1 / (2 * n_dim)
    shrink_factor = 1 - 1 / n_dim
    return reflection_factor, expansion_factor, contraction_factor, shrink_factor


class NelderMead(Optimizer):
    """
    Nelder-Mead optimizer.

    In this method we generate initial n_dim + 1 test points. We evaluate cost of every test point.

    After that, in every iteration, we generate reflection point. Direction of reflection goes through worst point and
    centroid of other points.

    Based on the reflection point cost fr and current test point costs (f1, f2, ..., fn, fn+1), we select one the
    following steps:
    - fr < f1: expansion or reflection, whichever is better
    - f1 <= fr < fn: reflection
    - fn <= fr < fn+1: outside contraction or shrink
    - fr >= fn+1: inside contraction or shrink

    In expansion, reflection and contraction, the worst point is replaced with new point. In shrink step, all the
    points are moved towards the best test point.
    """

    def __init__(self,
                 f_cost: Callable,
                 f_points: Callable = generate_init_test_points,
                 reflection_factor: float = 1.0,
                 expansion_factor: float = 2.0,
                 contraction_factor: float = 0.5,
                 shrink_factor: float = 0.5,
                 termination_checks=TERMINATION_CHECKS
                 ):
        """

        @param f_cost: See Optimizer.
        @param f_points: Function to generate initial test points from init guess: test_points = f_points(init_guess).
        @param reflection_factor: Step size to generate reflected point.
        @param expansion_factor: Step size to generate expanded point (> 1).
        @param contraction_factor: Step size to generate contracted point (< 1).
        @param shrink_factor: Step size to shrink points towards best point (< 1).
        @param termination_checks: See Optimizer.
        """
        super().__init__(f_cost, termination_checks)
        self.f_points = f_points
        self.reflection_factor = reflection_factor
        self.expansion_factor = expansion_factor
        self.contraction_factor = contraction_factor
        self.shrink_factor = shrink_factor
        self.points = None

    def update(self, param, iter_round, cost) -> Tuple[np.ndarray, float]:

        # At first iteration, generate initial test points
        if self.points is None:
            self._generate_init_test_points(param)

        # Centroid of test points, worst point not included
        centroid = np.mean(self.points[:-1], axis=0)

        # Generate reflected point
        worst_point = self.points[-1]
        reflected_point = centroid + self.reflection_factor * (centroid - worst_point)
        cost_reflected = self.f_cost(reflected_point)

        # Limits
        cost_best = self.point_costs[0]
        cost_second_worst = self.point_costs[-2]
        cost_worst = self.point_costs[-1]

        # Select case based on reflection cost and test point costs
        if cost_reflected < cost_best:
            self._expansion_or_reflection(centroid, reflected_point, cost_reflected)
        elif (cost_reflected >= cost_best) and (cost_reflected < cost_second_worst):
            self._reflection(reflected_point, cost_reflected)
        elif (cost_reflected >= cost_second_worst) and (cost_reflected < cost_worst):
            self._outside_contraction_or_shrink(centroid, reflected_point, cost_reflected)
        else:
            self.inside_contraction_or_shrink(centroid, reflected_point, cost_worst)

        # Sort based on costs
        indices = np.argsort(self.point_costs)
        self.points = self.points[indices]
        self.point_costs = self.point_costs[indices]

        return self.points[0], self.point_costs[0]

    def _generate_init_test_points(self, param):
        self.points = self.f_points(param)
        self.point_costs = np.array([self.f_cost(p) for p in self.points])
        indices = np.argsort(self.point_costs)
        self.points = self.points[indices]
        self.point_costs = self.point_costs[indices]

    def _expansion_or_reflection(self, centroid, reflected_point, cost_reflected):
        expanded_point = centroid + self.expansion_factor * (reflected_point - centroid)
        cost_expanded = self.f_cost(expanded_point)
        if cost_expanded < cost_reflected:
            logger.debug("Expansion")
            self.point_costs[-1] = cost_expanded
            self.points[-1] = expanded_point
        else:
            logger.debug("Reflection")
            self.point_costs[-1] = cost_reflected
            self.points[-1] = reflected_point

    def _reflection(self, reflected_point, cost_reflected):
        logger.debug("Reflection")
        self.point_costs[-1] = cost_reflected
        self.points[-1] = reflected_point

    def _outside_contraction_or_shrink(self, centroid, reflected_point, cost_reflected):
        point_contracted = centroid + self.contraction_factor * (reflected_point - centroid)
        cost_contracted = self.f_cost(point_contracted)
        if cost_contracted <= cost_reflected:
            logger.debug("Outside contraction")
            self.point_costs[-1] = cost_contracted
            self.points[-1] = point_contracted
        else:
            self._shrink()

    def inside_contraction_or_shrink(self, centroid, reflected_point, cost_worst):
        point_contracted = centroid - self.contraction_factor * (reflected_point - centroid)
        cost_contracted = self.f_cost(point_contracted)
        if cost_contracted <= cost_worst:
            logger.debug("Inside contraction")
            self.point_costs[-1] = cost_contracted
            self.points[-1] = point_contracted
        else:
            self._shrink()

    def _shrink(self):
        logger.debug("Shrink")
        self.points[1:] = self.points[0] + self.shrink_factor * (self.points[1:] - self.points[0])
