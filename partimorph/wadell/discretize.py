"""Boundary discretization and concave/convex classification."""

import numpy as np


def discretize_boundary(
    boundary: np.ndarray,
    max_dev_thresh: float,
) -> np.ndarray:
    """Discretize boundary into keypoints based on max deviation from line.

    Args:
        boundary: (N, 2) boundary points in (x, y).
        max_dev_thresh: Maximum allowed deviation from a straight line.

    Returns:
        (M, 2) keypoints filtered from the boundary.
    """

    # Swap to (y, x) to mirror wadell_rs-main pipeline
    x = boundary[:, 1]
    y = boundary[:, 0]
    n = len(x)

    seg_start = 0
    seg_end = n

    keypoints = [[x[seg_start], y[seg_start]]]

    # Append first point to close the boundary
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    total_len = len(x)  # n + 1

    while seg_start < n:
        max_dev, pos_idx = _maxlinedev(x[seg_start:seg_end], y[seg_start:seg_end])

        while max_dev > max_dev_thresh:
            new_end = pos_idx + seg_start
            # Guard: ensure progress
            if new_end <= seg_start:
                seg_end = seg_start + 2
                break
            seg_end = new_end
            max_dev, pos_idx = _maxlinedev(x[seg_start:seg_end], y[seg_start:seg_end])

        if seg_end != total_len or seg_start == 0:
            idx = min(seg_end - 1, total_len - 1)
            keypoints.append([x[idx], y[idx]])

        seg_start = seg_end
        seg_end = total_len

    return np.array(keypoints)


def classify_concave_convex(
    keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify keypoints into concave and convex.

    Shifts keypoints to start from the first concave point
    to avoid circle fitting issues at the boundary wrap.

    Args:
        keypoints: (M, 2) array of keypoints.

    Returns:
        (concave_points, convex_points) each as (K, 2) arrays.
    """

    kp = np.vstack([keypoints[-1], keypoints, keypoints[0]])

    v1 = kp[1:-1] - kp[:-2]
    v2 = kp[2:] - kp[1:-1]

    angle = np.arctan2(
        v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0],
        v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1],
    )

    area = _signed_area(keypoints)
    orientation = 1.0 if area >= 0.0 else -1.0
    oriented = angle * orientation

    concave_locs = np.where(oriented < 0.0)[0]
    roll_idx = concave_locs[0] if len(concave_locs) > 0 else int(np.argmax(oriented))

    kp = np.roll(keypoints, -roll_idx, axis=0)
    oriented = np.roll(oriented, -roll_idx)

    return kp[oriented < 0.0], kp[oriented > 0.0]


def _signed_area(points: np.ndarray) -> float:
    """Signed polygon area (shoelace). Positive for CCW."""

    x = points[:, 0]
    y = points[:, 1]

    # Avoids memory allocation overhead from np.roll by slicing
    area = np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0]

    return 0.5 * float(area)


def _maxlinedev(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, int]:
    """Find point with max deviation from line connecting first and last."""

    if len(x) <= 1:
        return 0.0, 0

    x0, x1 = x[0], x[-1]
    y0, y1 = y[0], y[-1]
    dx = x1 - x0
    dy = y0 - y1

    dist_end_sq = dx * dx + dy * dy

    if dist_end_sq < 1e-12:
        dx_arr = x - x0
        dy_arr = y - y0
        dist_sq = dx_arr * dx_arr + dy_arr * dy_arr
        idx = int(np.argmax(dist_sq))
        return float(np.sqrt(dist_sq[idx])), idx
    else:
        c = y1 * x0 - y0 * x1
        dist_num = np.abs(dy * x + dx * y + c)
        idx = int(np.argmax(dist_num))
        return float(dist_num[idx] / np.sqrt(dist_end_sq)), idx
