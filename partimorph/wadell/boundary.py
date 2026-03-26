import numpy as np
from skimage.measure import label, regionprops


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float64)

    labeled = label(mask, connectivity=2)
    properties = regionprops(labeled)

    if not properties:
        return np.empty((0, 2), dtype=np.float64)

    region = max(properties, key=lambda r: r.area)
    boundary_yx = boundary_tracing(region)

    if boundary_yx.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    if not np.array_equal(boundary_yx[0], boundary_yx[-1]):
        boundary_yx = np.vstack([boundary_yx, boundary_yx[0]])

    boundary_xy = np.flip(boundary_yx, axis=1).astype(np.float64)

    return boundary_xy


def moore_neighborhood(current: np.ndarray, backtrack: np.ndarray) -> np.ndarray | int:
    operations = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    )
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            return np.concatenate((neighbors[i:], neighbors[:i]))

    return 0


def boundary_tracing(region) -> np.ndarray:
    coords = region.coords
    max_coords = np.amax(coords, axis=0)

    binary = np.zeros((max_coords[0] + 2, max_coords[1] + 2), dtype=np.uint8)
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1

    start_index = 0

    while True:
        start = [y[start_index], x[start_index]]
        focus = binary[start[0] - 1 : start[0] + 2, start[1] - 1 : start[1] + 2]

        if np.sum(focus) > 1:
            break

        start_index += 1
        if start_index >= len(x):
            return np.empty((0, 2), dtype=np.int64)

    if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = np.array(start)
    backtrack = np.array(backtrack_start)
    boundary = []

    while True:
        neighbors = moore_neighborhood(current, backtrack)

        if isinstance(neighbors, int):
            return np.empty((0, 2), dtype=np.int64)

        neighbor_y = neighbors[:, 0]
        neighbor_x = neighbors[:, 1]
        idx = int(np.argmax(binary[tuple([neighbor_y, neighbor_x])]))

        boundary.append(current.copy())
        backtrack = neighbors[idx - 1]
        current = neighbors[idx]

        if np.all(current == start) and np.all(backtrack == backtrack_start):
            break

    return np.array(boundary, dtype=np.int64)
