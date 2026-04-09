import numpy as np


def smooth_boundary(
    points: np.ndarray,
    perimeter: float,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> np.ndarray:
    coordinates = points[:-1]
    num_points = len(coordinates)

    alpha = alpha_ratio * perimeter
    beta = beta_ratio * perimeter
    regularization_mat = regularization_matrix(num_points, alpha, beta)
    result, *_ = np.linalg.lstsq(regularization_mat, coordinates, rcond=None)
    return result


def regularization_matrix(num_points: int, alpha: float, beta: float) -> np.ndarray:
    coefficients = alpha * np.array([-2, 1, 0, 0]) + beta * np.array([-6, 4, -1, 0])

    distance_matrix = np.fromfunction(
        lambda i, j: np.minimum((i - j) % num_points, (j - i) % num_points),
        (num_points, num_points),
        dtype=int,
    )

    A = coefficients[np.minimum(distance_matrix, len(coefficients) - 1)]
    return np.eye(num_points) - A
