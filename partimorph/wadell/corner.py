import numpy as np
from scipy.linalg import lstsq


def compute_corner_circles(
    convex_points: np.ndarray,
    keypoints: np.ndarray,
    r_max: float,
    r_max_pos: np.ndarray,
    circle_fit_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_points = len(convex_points)

    radii: list[float] = []
    centers: list[np.ndarray] = []
    start = 0

    while start < num_points - 1:
        for end in range(num_points, start + 2, -1):
            point_subset = convex_points[start:end]
            radius, center = nsphere_fit(point_subset)

            if radius <= 1e-06 or not np.isfinite(radius):
                continue

            dist_center = np.linalg.norm(center - r_max_pos)
            dist_boundary = np.linalg.norm(point_subset - r_max_pos, axis=1).mean()

            if dist_center < dist_boundary and radius < r_max:
                dist_all = np.linalg.norm(keypoints - center, axis=1)

                if np.all(dist_all / radius > circle_fit_thresh):
                    radii.append(radius)
                    centers.append(center)
                    start = end - 1
                    break

        start += 1

    if len(radii) == 0:
        return (np.array([]), np.empty((0, 2)))

    return (np.array(radii), np.array(centers))


def nsphere_fit(x: np.ndarray) -> tuple[float, np.ndarray]:
    points = x.reshape(-1, 2).copy()
    num_pts = points.shape[0]

    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    x_span = x_max - x_min
    y_span = y_max - y_min
    scale = 0.5 * max(x_span, y_span)

    if scale < 1e-12:
        return (0.0, points[0].copy())

    offset = np.array([(x_max + x_min) * 0.5, (y_max + y_min) * 0.5], dtype=np.float64)
    points_norm = (points - offset) / scale

    B = np.empty((num_pts, 3), dtype=points_norm.dtype)
    B[:, :2] = points_norm
    B[:, 2] = 1.0

    d = np.square(points_norm).sum(axis=1)
    y, *_ = lstsq(B, d, overwrite_a=True, overwrite_b=True)

    norm_center = 0.5 * y[:2]
    norm_radius = float(np.sqrt(y[2] + np.square(norm_center).sum()))
    center = norm_center * scale + offset

    if not np.all(np.isfinite(center)):
        return (0.0, points[0].copy())

    radius = norm_radius * scale

    if not np.isfinite(radius):
        return (0.0, center)

    return (radius, center)
