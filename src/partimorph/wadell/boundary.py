import cv2
import numpy as np


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2).astype(np.float64)

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    return points
