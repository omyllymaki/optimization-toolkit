import logging
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.levenberg_marquardt import LevenbergMarquardt
from src.termination import check_n_iter, check_absolute_cost_diff
from src.utils import mse

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

NOISE = 0.05


def coeff_to_transform_matrix(coeff):
    ax, ay, az, tx, ty, tz = coeff
    rx = R.from_euler('x', ax).as_matrix()
    ry = R.from_euler('y', ay).as_matrix()
    rz = R.from_euler('z', az).as_matrix()
    rot_mat = rx @ ry @ rz
    trans = np.array([tx, ty, tz])

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_mat
    transform_matrix[:3, 3] = trans

    return transform_matrix


def transform(transform_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    xyz1 = np.hstack((points, ones))
    transformed = (transform_matrix @ xyz1.T).T
    return transformed[:, 0:3]


def f_eval(xyz, coeff):
    t = coeff_to_transform_matrix(coeff)
    return transform(t, xyz)


def calculate_distances(source, target):
    diff = source - target
    return np.linalg.norm(diff, axis=1)


def f_cost(param, source, target):
    source_transformed = f_eval(source, param)
    errors = calculate_distances(source_transformed, target)
    return mse(errors)


def f_err(param, source, target):
    source_transformed = f_eval(source, param)
    return calculate_distances(source_transformed, target)


def main():
    for k in range(9):
        source = np.random.randn(50, 3)
        param_true = np.random.randn(6)
        target = f_eval(source, param_true) + NOISE * np.random.randn(50, 3)

        init_guess = np.zeros(6)
        termination_checks = (
            partial(check_n_iter, threshold=500),
            partial(check_absolute_cost_diff, threshold=1e-9)
        )
        fe = partial(f_err, source=source, target=target)
        optimizer = LevenbergMarquardt(f_err=fe, termination_checks=termination_checks)
        t1 = time.time()
        param, costs, _ = optimizer.run(init_guess)
        t2 = time.time()
        duration_ms = 1000 * (t2 - t1)
        t = coeff_to_transform_matrix(param)
        source_transformed = transform(t, source)

        print(f"Test {k}")
        print(f"True param: {param_true}")
        print(f"Estimated param: {param}")

        distances = calculate_distances(source_transformed, target)
        mean_distance = np.mean(distances)

        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.cla()
        plt.plot(source[:, 0], source[:, 1], "bo", alpha=0.5)
        plt.plot(target[:, 0], target[:, 1], "ro", alpha=0.5)

        plt.subplot(2, 2, 2)
        plt.cla()
        plt.plot(source_transformed[:, 0], source_transformed[:, 1], "bo", alpha=0.5)
        plt.plot(target[:, 0], target[:, 1], "ro", alpha=0.5)

        plt.subplot(2, 1, 2)
        plt.cla()
        plt.plot(costs, "-o")

        plt.figure(2)
        plt.subplot(3, 3, k + 1)
        plt.plot(source_transformed[:, 0], source_transformed[:, 1], "b.", alpha=0.5)
        plt.plot(target[:, 0], target[:, 1], "r.", alpha=0.5)
        plt.title(f"{mean_distance:0.2f} | {duration_ms:0.1f} ms")

        plt.pause(0.1)

    plt.show()


if __name__ == "__main__":
    main()
