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
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask, _, _ = crop_mask(mask, pad=1)

    if mask.size == 0:
        return None

    dist = ndimage.distance_transform_edt(mask.astype(bool))
    r_max = float(np.max(dist))

    if r_max < 1e-06:
        return None

    idx = np.argmax(dist)
    max_y, max_x = np.unravel_index(idx, dist.shape)
    r_max_pos = np.array([float(max_y), float(max_x)])

    boundary = extract_boundary(mask)

    if len(boundary) < 4:
        return None

    perimeter = float(
        cv2.arcLength(boundary[:-1].astype(np.float32).reshape(-1, 1, 2), closed=True)
    )

    if perimeter < 1e-06:
        return None

    smoothed = smooth_boundary(
        boundary, perimeter=perimeter, alpha_ratio=alpha_ratio, beta_ratio=beta_ratio
    )

    keypoints = discretize_boundary(smoothed, max_dev_thresh)

    if len(keypoints) < 3:
        return None

    _, convex_points = classify_concave_convex(keypoints)

    if len(convex_points) < 2:
        return None

    radii, _ = compute_corner_circles(
        convex_points, keypoints, r_max, r_max_pos, circle_fit_thresh
    )

    if len(radii) == 0:
        return None

    return float(np.mean(radii) / r_max)
