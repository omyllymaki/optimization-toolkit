import logging
from typing import Callable, Tuple

import numpy as np

from src.optimizer import Optimizer
from src.termination import TerminationCriteria as TC

logger = logging.getLogger(__name__)


def generate_init_test_points(param, scale):
    """
    Helper function generate test points for Nelder-Mead optimizer.
    """
    n_dim = len(param)
    points = param.copy()
    for k in range(n_dim):
        v = np.random.randn(len(points))
        v = v / np.linalg.norm(v)
        point = param + scale * v
        points = np.vstack((points, point))
    return points


class NelderMead(Optimizer):
    """
    Nelder-Mead optimizer.

    In this method we generate n_dim + 1 test points. We evaluate cost of every test point. Then the worst (highest
    cost) test point is replaced with new test point in every iteration using following rules:

    Generate reflected point
    Case 1:  Reflected point is better than the second worst, but not better than the best test point
        -> Replace worst point with reflected point
    Case 2: Reflected point is the better than any test point
        -> Generate expanded point
        -> Replace worst point with expanded point or reflected point, whichever is better
    Case 3: Reflected point is the worse than second worst test point
        -> Generate contracted point
        -> If contracted point is better than worst test point, replace it with contracted point
        -> Else shrink all test point towards best test point (this should happen only in some rare situations)

    reflected_point = worst_point + reflection_factor * update_direction; reflection_factor > 1
    expanded_point = worst_point + expansion_factor * update_direction; expansion_factor > reflection_factor
    point_contracted = worst_point + contraction_factor * update_direction; contraction_factor < 1.0

    Update direction is vector from worst test point to centroid of other test points.
    """

    def __init__(self,
                 f_cost: Callable,
                 init_test_points: np.ndarray,
                 reflection_factor=2.0,
                 expansion_factor=4.0,
                 contraction_factor=0.5,
                 shrink_factor=0.5,
                 termination=TC(max_iter=1000,
                                max_iter_without_improvement=200,
                                cost_threshold=1e-6,
                                cost_diff_threshold=-np.inf)
                 ):
        """

        @param f_cost: See Optimizer.
        @param init_test_points: Initial test points (n_param + 1 points)
        @param reflection_factor: Step size to generate reflected point; should be > 1.
        @param expansion_factor: Step size to generate expanded point: should be > reflection_factor.
        @param contraction_factor: Step size to generate contracted point: should be < 1.
        @param shrink_factor: Step size to shrink points towards best point; should be < 1.
        @param termination: See Optimizer.
        """
        super().__init__(f_cost, termination)
        self.init_test_points = init_test_points
        self.reflection_factor = reflection_factor
        self.expansion_factor = expansion_factor
        self.contraction_factor = contraction_factor
        self.shrink_factor = shrink_factor

        self.points = init_test_points.copy()
        self.point_costs = np.array([self.f_cost(p) for p in self.points])
        indices = np.argsort(self.point_costs)
        self.points = self.points[indices]
        self.point_costs = self.point_costs[indices]

    def update(self, param, iter_round, cost) -> Tuple[np.ndarray, float]:

        # Centroid of test points, except worst
        centroid = np.mean(self.points[:-1], axis=0)

        # Get direction for update
        worst_point = self.points[-1]
        update_direction = centroid - worst_point

        # Generate reflected point
        reflected_point = worst_point + self.reflection_factor * update_direction
        cost_reflected = self.f_cost(reflected_point)

        # Case 1: Reflected point is better than the second worst, but not better than the best test point
        if (cost_reflected < self.point_costs[-2]) and (cost_reflected > self.point_costs[0]):
            self.point_costs[-1] = cost_reflected
            self.points[-1] = reflected_point
            logger.info("Reflection, not best so far")

        # Case 2: Reflected point is better than any test point
        # Try expansion
        elif cost_reflected < self.point_costs[0]:
            expanded_point = worst_point + self.expansion_factor * update_direction
            cost_expanded = self.f_cost(expanded_point)
            if cost_expanded < cost_reflected:
                self.point_costs[-1] = cost_expanded
                self.points[-1] = expanded_point
                logger.debug("Expansion, best so far")
            else:
                self.point_costs[-1] = cost_reflected
                self.points[-1] = reflected_point
                logger.debug("Reflection, best so far")

        # Case 3: Reflected point is worse than second worst test point
        # Try contraction
        elif cost_reflected > self.point_costs[-2]:
            point_contracted = worst_point + self.contraction_factor * update_direction
            cost_contracted = self.f_cost(point_contracted)

            # Contracted point is better than worst point
            if cost_contracted < self.point_costs[-1]:
                self.point_costs[-1] = cost_contracted
                self.points[-1] = point_contracted
                logger.debug("Contraction")
            # Shrink polygon: move all points towards best point
            else:
                self.points[1:] = self.points[0] + self.shrink_factor * (self.points[1:] - self.points[0])
                logger.debug("Shrink")

        # Sort based on costs
        indices = np.argsort(self.point_costs)
        self.points = self.points[indices]
        self.point_costs = self.point_costs[indices]

        return self.points[0], self.point_costs[0]
