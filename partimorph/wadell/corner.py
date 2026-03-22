import numpy as np
from scipy.linalg import lstsq


def compute_corner_circles(
    convex_points: np.ndarray,
    keypoints: np.ndarray,
    r_max: float,
    r_max_pos: np.ndarray,
    circle_fit_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(convex_points)

    radii: list[float] = []
    centers: list[np.ndarray] = []
    start = 0

    while start < n - 1:
        for end in range(n, start + 2, -1):
            pts_slice = convex_points[start:end]
            radius, center = _nsphere_fit(pts_slice)

            if radius <= 1e-06 or not np.isfinite(radius):
                continue

            dist_center = np.linalg.norm(center - r_max_pos)
            dist_boundary = np.linalg.norm(pts_slice - r_max_pos, axis=1).mean()

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


def _nsphere_fit(x: np.ndarray) -> tuple[float, np.ndarray]:
    pts = x.reshape(-1, 2).copy()
    m = pts.shape[0]

    x_min = float(pts[:, 0].min())
    x_max = float(pts[:, 0].max())
    y_min = float(pts[:, 1].min())
    y_max = float(pts[:, 1].max())

    x_span = x_max - x_min
    y_span = y_max - y_min
    scale = 0.5 * max(x_span, y_span)

    if scale < 1e-12:
        return (0.0, pts[0].copy())

    offset = np.array([(x_max + x_min) * 0.5, (y_max + y_min) * 0.5], dtype=np.float64)
    pts_norm = (pts - offset) / scale

    B = np.empty((m, 3), dtype=pts_norm.dtype)
    B[:, :2] = pts_norm
    B[:, 2] = 1.0

    d = np.square(pts_norm).sum(axis=1)
    y, *_ = lstsq(B, d, overwrite_a=True, overwrite_b=True)

    center_norm = 0.5 * y[:2]
    r_norm = float(np.sqrt(y[2] + np.square(center_norm).sum()))
    center = center_norm * scale + offset

    if not np.all(np.isfinite(center)):
        return (0.0, pts[0].copy())

    r = r_norm * scale

    if not np.isfinite(r):
        return (0.0, center)

    return (r, center)
