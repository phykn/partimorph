import numpy as np


def smooth_boundary(
    pts: np.ndarray,
    perimeter: float,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> np.ndarray:
    coords = pts[:-1]
    n = len(coords)

    alpha = alpha_ratio * perimeter
    beta = beta_ratio * perimeter

    reg_matrix = _regularization_matrix(n, alpha, beta)
    smoothed = np.linalg.solve(reg_matrix, coords)

    return smoothed


def _regularization_matrix(n: int, alpha: float, beta: float) -> np.ndarray:
    d = alpha * np.array([-2, 1, 0, 0]) + beta * np.array([-6, 4, -1, 0])

    D = np.fromfunction(
        lambda i, j: np.minimum((i - j) % n, (j - i) % n), (n, n), dtype=int
    )

    A = d[np.minimum(D, len(d) - 1)]

    return np.eye(n) - A
