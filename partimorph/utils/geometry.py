import cv2
import numpy as np


def create_poly_mask(
    shape: tuple[int, int],
    vertices: np.ndarray,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    res = np.zeros(shape, dtype=dtype)
    pts = vertices.astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(res, [pts], 1)
    return res


def polar_vertices(
    center: tuple[float, float],
    radii: float | np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    cy, cx = center
    radii_arr = np.asarray(radii, dtype=float)
    return np.stack(
        [cx + radii_arr * np.cos(angles), cy + radii_arr * np.sin(angles)],
        axis=1,
    )
