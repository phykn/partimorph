import numpy as np
from .geometry import create_poly_mask, polar_vertices


def create_circle_mask(
    shape: tuple[int, int], center: tuple[int, int], radius: float
) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be >= 0.")

    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center

    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def create_ellipse_mask(
    shape: tuple[int, int], center: tuple[int, int], radius_x: float, radius_y: float
) -> np.ndarray:
    if radius_x <= 0 or radius_y <= 0:
        raise ValueError("radius_x and radius_y must be > 0.")

    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center

    return ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2 <= 1.0


def create_rectangle_mask(
    shape: tuple[int, int], top_left: tuple[int, int], bottom_right: tuple[int, int]
) -> np.ndarray:
    res = np.zeros(shape, dtype=bool)

    h, w = shape
    y1, x1 = top_left
    y2, x2 = bottom_right

    if y1 >= y2 or x1 >= x2:
        raise ValueError("top_left must be strictly above-left of bottom_right.")
    if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
        raise ValueError("rectangle coordinates must be within mask bounds.")

    res[y1:y2, x1:x2] = True

    return res


def create_square_mask(
    shape: tuple[int, int], top_left: tuple[int, int], size: int
) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be > 0.")

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

    return create_poly_mask(shape, vertices).astype(bool)


def create_pentagon_mask(
    shape: tuple[int, int], center: tuple[int, int], radius: float
) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi / 2
    vertices = polar_vertices(center=center, radii=radius, angles=angles)

    return create_poly_mask(shape, vertices).astype(bool)


def create_star_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    outer_radius: float,
    inner_radius: float,
    num_points: int = 5,
) -> np.ndarray:
    if outer_radius <= 0:
        raise ValueError("outer_radius must be > 0.")
    if inner_radius <= 0:
        raise ValueError("inner_radius must be > 0.")
    if not isinstance(num_points, (int, np.integer)):
        raise TypeError("num_points must be an integer.")
    if num_points < 2:
        raise ValueError("num_points must be >= 2.")

    angles = np.linspace(0, 2 * np.pi, 2 * num_points + 1)[:-1] - np.pi / 2
    radii = np.ones_like(angles) * outer_radius
    radii[1::2] = inner_radius

    vertices = polar_vertices(center=center, radii=radii, angles=angles)

    return create_poly_mask(shape, vertices).astype(bool)
