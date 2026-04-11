import cv2
import numpy as np

from .boundary import extract_boundary
from .corner import compute_corner_circles
from .discretize import classify_concave_convex, discretize_boundary
from .smoothing import smooth_boundary
from ..misc import crop_mask
from ..schema import Mask, Points


def compute_roundness(
    mask: Mask,
    max_dev_thresh: float = 0.3,
    circle_fit_thresh: float = 0.98,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> float | None:
    mask_cropped, _, _ = crop_mask(mask, pad=1)
    if mask_cropped.size == 0:
        return None

    distance_transform = cv2.distanceTransform(
        mask_cropped, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    max_radius = float(np.max(distance_transform))
    if max_radius < 1e-6:
        return None

    max_idx = np.argmax(distance_transform)
    peak_y, peak_x = np.unravel_index(max_idx, distance_transform.shape)
    max_radius_pos = np.array([float(peak_x), float(peak_y)])

    boundary = extract_boundary(mask_cropped)
    if len(boundary) < 4:
        return None

    perimeter = float(
        cv2.arcLength(boundary[:-1].astype(np.float32).reshape(-1, 1, 2), closed=True)
    )
    if perimeter < 1e-6:
        return None

    smoothed_boundary = smooth_boundary(
        boundary,
        perimeter=perimeter,
        alpha_ratio=alpha_ratio,
        beta_ratio=beta_ratio,
    )

    keypoints = discretize_boundary(smoothed_boundary, max_dev_thresh)
    if len(keypoints) < 3:
        return None

    _, convex_points = classify_concave_convex(keypoints)
    if len(convex_points) < 2:
        return None

    convex_points_typed: Points = convex_points.astype(np.float32)

    radii, _ = compute_corner_circles(
        convex_points_typed,
        keypoints.astype(np.float32),
        max_radius,
        max_radius_pos,
        circle_fit_thresh,
    )
    if len(radii) == 0:
        return None

    return float(np.mean(radii) / max_radius)
