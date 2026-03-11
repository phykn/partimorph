"""Boundary extraction from binary mask using cv2.findContours."""

import cv2
import numpy as np


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract ordered boundary points from a binary mask.

    Args:
        mask: Binary mask (0/1, uint8).

    Returns:
        Boundary points as (N, 2) float64 array in (x, y) format.
        The contour is closed (first point appended at end).
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    # Use the largest contour
    cnt = max(contours, key=len)
    pts = cnt.squeeze(axis=1).astype(np.float64)

    # Close the contour
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    return pts
