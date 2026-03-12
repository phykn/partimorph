"""Curve smoothing via energy minimization (active contours / snakes).

References:
    [1] Kass, M., et al. "Snakes - Active Contour Models."
        International Journal of Computer Vision, vol. 1, no. 4, 1987, pp. 321-31.
    [2] Xu, C., Pham, D. & Prince, J. (2000). Image Segmentation Using Deformable Models.
"""

import numpy as np


def smooth_boundary(
    pts: np.ndarray,
    perimeter: float,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> np.ndarray:
    """Smooth closed boundary points using energy minimization.

    Args:
        pts: (N+1, 2) boundary points where pts[0] == pts[-1] (closed).
        perimeter: Perimeter length for scaling regularization weights.
        alpha_ratio: Elasticity weight ratio.
        beta_ratio: Rigidity weight ratio.

    Returns:
        Smoothed (N, 2) boundary points (open, not repeated).
    """

    coords = pts[:-1]
    n = len(coords)

    alpha = alpha_ratio * perimeter
    beta = beta_ratio * perimeter

    inv_matrix = _regularization_matrix(n, alpha, beta)
    smoothed = np.matmul(inv_matrix, coords)

    return smoothed


def _regularization_matrix(
    n: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """An NxN matrix for imposing elasticity and rigidity to snakes."""

    d = alpha * np.array([-2, 1, 0, 0]) + beta * np.array([-6, 4, -1, 0])

    # Distance modulo matrix
    D = np.fromfunction(
        lambda i, j: np.minimum((i - j) % n, (j - i) % n),
        (n, n),
        dtype = int,
    )

    A = d[np.minimum(D, len(d) - 1)]
    return np.linalg.inv(np.eye(n) - A)
