"""Boundary extraction from binary mask using Moore neighborhood tracing."""

import numpy as np
from skimage.measure import label, regionprops


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract ordered boundary points from a binary mask.

    Args:
        mask: Binary mask (0/1, uint8).

    Returns:
        Boundary points as (N, 2) float64 array in (x, y) format.
        The contour is closed (first point appended at end).
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float64)

    lbl = label(mask, connectivity=2)
    props = regionprops(lbl)
    if not props:
        return np.empty((0, 2), dtype=np.float64)

    region = max(props, key=lambda r: r.area)
    boundary_yx = _boundary_tracing(region)

    if boundary_yx.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Close the contour
    if not np.array_equal(boundary_yx[0], boundary_yx[-1]):
        boundary_yx = np.vstack([boundary_yx, boundary_yx[0]])

    # Convert (y, x) -> (x, y)
    boundary_xy = np.flip(boundary_yx, axis=1).astype(np.float64)
    return boundary_xy


def _moore_neighborhood(
    current: np.ndarray,
    backtrack: np.ndarray,
) -> np.ndarray | int:
    """Clockwise list of pixels from Moore neighborhood of current pixel."""

    operations = np.array(
        [
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
        ]
    )
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0


def _boundary_tracing(region) -> np.ndarray:
    """Coordinates of the region's boundary in clockwise order.

    Returns:
        (N, 2) array of (y, x) boundary coordinates.
    """

    coords = region.coords
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] + 2, maxs[1] + 2), dtype=np.uint8)
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1

    # Find the most upper-left non-isolated pixel
    idx_start = 0
    while True:
        start = [y[idx_start], x[idx_start]]
        focus = binary[start[0] - 1 : start[0] + 2, start[1] - 1 : start[1] + 2]
        if np.sum(focus) > 1:
            break
        idx_start += 1
        if idx_start >= len(x):
            return np.empty((0, 2), dtype=np.int64)

    if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = np.array(start)
    backtrack = np.array(backtrack_start)
    boundary = []

    while True:
        neighbors = _moore_neighborhood(current, backtrack)
        if isinstance(neighbors, int):
            return np.empty((0, 2), dtype=np.int64)
        ny = neighbors[:, 0]
        nx = neighbors[:, 1]
        idx = int(np.argmax(binary[tuple([ny, nx])]))
        boundary.append(current.copy())
        backtrack = neighbors[idx - 1]
        current = neighbors[idx]

        if np.all(current == start) and np.all(backtrack == backtrack_start):
            break

    return np.array(boundary, dtype=np.int64)
