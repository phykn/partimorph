"""Wadell roundness calculation from a binary mask.

Implements the Zheng & Hryciw (2015) algorithm:
    boundary extraction → smoothing → discretization →
    concave/convex classification → corner circle fitting →
    roundness = mean(corner_radii) / R_max

Reference:
    Zheng, J., and R. D. Hryciw. "Traditional Soil Particle Sphericity,
    Roundness and Surface Roughness by Computational Geometry."
    Geotechnique, vol. 65, no. 6, 2015, pp. 494-506.
"""

import cv2
import numpy as np
from scipy import ndimage

from .boundary import extract_boundary
from .corner import compute_corner_circles
from .discretize import classify_concave_convex, discretize_boundary
from .smoothing import smooth_boundary
from ..misc import crop_mask


def compute_roundness(
    mask: np.ndarray,
    max_dev_thresh: float = 0.3,
    circle_fit_thresh: float = 0.98,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> float:
    """Compute Wadell roundness from a binary mask.

    Args:
        mask: Binary mask (values 0/1, dtype uint8).
        max_dev_thresh: Max deviation threshold for boundary discretization.
        circle_fit_thresh: Circle fit validation threshold (0-1).
        alpha_ratio: Elasticity weight for boundary smoothing.
        beta_ratio: Rigidity weight for boundary smoothing.

    Returns:
        Roundness value (float). Returns 0.0 for invalid inputs.
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask, _, _ = crop_mask(mask, pad=1)

    if mask.size == 0:
        return None

    # --- Maximum inscribed circle (R_max) ---
    dist = ndimage.distance_transform_edt(mask.astype(bool))
    r_max = float(np.max(dist))

    if r_max < 1e-6:
        return None

    idx = np.argmax(dist)
    max_y, max_x = np.unravel_index(idx, dist.shape)
    # Keep (y, x) ordering to match discretization and corner fitting.
    r_max_pos = np.array([float(max_y), float(max_x)])

    # --- Boundary extraction ---
    boundary = extract_boundary(mask)

    if len(boundary) < 4:
        return None

    # --- Perimeter for smoothing weights ---
    perimeter = float(
        cv2.arcLength(
            boundary[:-1].astype(np.float32).reshape(-1, 1, 2),
            closed = True,
        )
    )

    if perimeter < 1e-6:
        return None

    # --- Smooth boundary ---
    smoothed = smooth_boundary(
        boundary,
        perimeter = perimeter,
        alpha_ratio = alpha_ratio,
        beta_ratio = beta_ratio,
    )

    # --- Discretize → concave/convex → corner circles ---
    keypoints = discretize_boundary(smoothed, max_dev_thresh)

    if len(keypoints) < 3:
        return None

    _, convex_points = classify_concave_convex(keypoints)

    if len(convex_points) < 2:
        return None

    radii, _ = compute_corner_circles(
        convex_points,
        keypoints,
        r_max,
        r_max_pos,
        circle_fit_thresh,
    )

    if len(radii) == 0:
        return None

    return float(np.mean(radii) / r_max)
