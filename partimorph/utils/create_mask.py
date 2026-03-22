import cv2
import numpy as np


def _create_poly_mask(shape: tuple[int, int], vertices: np.ndarray) -> np.ndarray:
    res = np.zeros(shape, dtype=np.uint8)
    pts = vertices.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(res, [pts], 1)
    return res.astype(bool)


def create_circle_mask(
    shape: tuple[int, int], center: tuple[int, int], radius: float
) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def create_ellipse_mask(
    shape: tuple[int, int], center: tuple[int, int], radius_x: float, radius_y: float
) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    return ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2 <= 1.0


def create_rectangle_mask(
    shape: tuple[int, int], top_left: tuple[int, int], bottom_right: tuple[int, int]
) -> np.ndarray:
    res = np.zeros(shape, dtype=bool)
    y1, x1 = top_left
    y2, x2 = bottom_right
    res[y1:y2, x1:x2] = True
    return res


def create_square_mask(
    shape: tuple[int, int], top_left: tuple[int, int], size: int
) -> np.ndarray:
    return create_rectangle_mask(
        shape=shape,
        top_left=top_left,
        bottom_right=(top_left[0] + size, top_left[1] + size),
    )


def create_triangle_mask(
    shape: tuple[int, int],
    v1: tuple[int, int],
    v2: tuple[int, int],
    v3: tuple[int, int],
) -> np.ndarray:
    vertices = np.array([v1, v2, v3])
    return _create_poly_mask(shape, vertices)


def create_pentagon_mask(
    shape: tuple[int, int], center: tuple[int, int], radius: float
) -> np.ndarray:
    cy, cx = center
    angles = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi / 2
    vertices = np.stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1
    )
    return _create_poly_mask(shape, vertices)


def create_star_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    outer_radius: float,
    inner_radius: float,
    num_points: int = 5,
) -> np.ndarray:
    cy, cx = center
    angles = np.linspace(0, 2 * np.pi, 2 * num_points + 1)[:-1] - np.pi / 2
    radii = np.ones_like(angles) * outer_radius
    radii[1::2] = inner_radius
    vertices = np.stack(
        [cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1
    )
    return _create_poly_mask(shape, vertices)
