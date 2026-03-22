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
    x = x.reshape(-1, 2).copy()
    m = x.shape[0]
    xmin = x.min()
    xmax = x.max()
    scale = 0.5 * (xmax - xmin)
    offset = 0.5 * (xmax + xmin)
    if scale < 1e-12:
        return (0.0, x[0].copy())
    x -= offset
    x /= scale
    B = np.empty((m, 3), dtype=x.dtype)
    B[:, :2] = x
    B[:, 2] = 1.0
    d = np.square(x).sum(axis=1)
    y, *_ = lstsq(B, d, overwrite_a=True, overwrite_b=True)
    c = 0.5 * y[:2]
    r = float(np.sqrt(y[2] + np.square(c).sum()))
    r *= scale
    c = c * scale + offset
    return (r, c)
