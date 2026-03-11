import cv2
import numpy as np
import skimage.measure

from scipy import ndimage

from .misc import crop_mask, get_contours
from .wadell import compute_roundness as _compute_roundness


def find_inscribed_circle(mask: np.ndarray) -> tuple[int, int, float]:

    cropped_mask, pad_x0, pad_y0 = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return 0, 0, 0.0

    dist = ndimage.distance_transform_edt(cropped_mask.astype(bool))

    idx = np.argmax(dist)
    cy_crop, cx_crop = np.unravel_index(idx, dist.shape)

    cx = int(cx_crop) + pad_x0
    cy = int(cy_crop) + pad_y0
    r = float(dist[cy_crop, cx_crop])

    return cx, cy, r


def find_enclosing_circle(mask: np.ndarray) -> tuple[int, int, float]:

    if not np.any(mask):
        return 0, 0, 0.0

    contours = get_contours(mask)

    pts = np.concatenate(contours)
    hull = cv2.convexHull(pts)
    (cx, cy), r = cv2.minEnclosingCircle(hull)

    x = int(round(cx))
    y = int(round(cy))
    val_r = float(r)

    return x, y, val_r


def fit_ellipse(mask: np.ndarray) -> dict | None:

    contours = get_contours(mask)

    if not contours:
        return None

    pts = np.concatenate(contours)

    if len(pts) < 5:
        return None

    (cx, cy), (w, h), angle = cv2.fitEllipse(pts)

    minor = float(min(w, h))
    major = float(max(w, h))
    ratio = major / minor if minor > 0 else float("inf")

    rect = ((cx, cy), (w, h), angle)
    bbox = cv2.boxPoints(rect).astype(np.float32)

    return {
        "major": major,
        "minor": minor,
        "ratio": ratio,
        "bbox": bbox,
    }


def compute_roundness(mask: np.ndarray) -> float:

    return _compute_roundness(mask)


def compute_circularity(mask: np.ndarray) -> float:

    cropped_mask, _, _ = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return 0.0

    perimeter = skimage.measure.perimeter_crofton(cropped_mask, 4)

    if perimeter <= 0.0:
        return 0.0

    area = float(np.count_nonzero(mask))

    val_circularity = 4.0 * np.pi * area / (perimeter ** 2)

    return val_circularity