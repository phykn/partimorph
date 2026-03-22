import cv2
import numpy as np
from scipy import ndimage
from typing import TypedDict
from .misc import crop_mask, get_contours


class CircleData(TypedDict):
    x: int
    y: int
    r: float


class EllipseData(TypedDict):
    major: float
    minor: float
    x: float
    y: float
    angle: float
    w: float
    h: float
    bbox: list[list[float]]


def find_inscribed_circle(mask: np.ndarray) -> CircleData | None:
    cropped_mask, pad_x0, pad_y0 = crop_mask(mask, pad=1)
    if cropped_mask.size == 0:
        return None
    dist = ndimage.distance_transform_edt(cropped_mask.astype(bool))
    idx = np.argmax(dist)
    cy_crop, cx_crop = np.unravel_index(idx, dist.shape)
    x = int(cx_crop) + pad_x0
    y = int(cy_crop) + pad_y0
    r = float(dist[cy_crop, cx_crop])
    return {"x": x, "y": y, "r": r}


def find_enclosing_circle(mask: np.ndarray) -> CircleData | None:
    if not np.any(mask):
        return None
    contours = get_contours(mask)
    if not contours:
        return None
    pts = np.concatenate(contours)
    hull = cv2.convexHull(pts)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    x = int(round(cx))
    y = int(round(cy))
    r = float(r)
    return {"x": x, "y": y, "r": r}


def fit_ellipse(mask: np.ndarray) -> EllipseData | None:
    contours = get_contours(mask)
    if not contours:
        return None
    pts = np.concatenate(contours).astype(np.float32)
    if len(pts) < 5:
        return None
    _, _, angle = cv2.fitEllipse(pts)
    rad = np.deg2rad(angle)
    cos_a, sin_a = (np.cos(rad), np.sin(rad))
    vec_w = np.array([cos_a, sin_a])
    vec_h = np.array([-sin_a, cos_a])
    proj_w = pts @ vec_w
    proj_h = pts @ vec_h
    min_w, max_w = (proj_w.min(), proj_w.max())
    min_h, max_h = (proj_h.min(), proj_h.max())
    w_tight = max_w - min_w
    h_tight = max_h - min_h
    mid_w = (max_w + min_w) / 2.0
    mid_h = (max_h + min_h) / 2.0
    cx = mid_w * vec_w[0] + mid_h * vec_h[0]
    cy = mid_w * vec_w[1] + mid_h * vec_h[1]
    bbox = [
        [
            float(min_w * vec_w[0] + min_h * vec_h[0]),
            float(min_w * vec_w[1] + min_h * vec_h[1]),
        ],
        [
            float(max_w * vec_w[0] + min_h * vec_h[0]),
            float(max_w * vec_w[1] + min_h * vec_h[1]),
        ],
        [
            float(max_w * vec_w[0] + max_h * vec_h[0]),
            float(max_w * vec_w[1] + max_h * vec_h[1]),
        ],
        [
            float(min_w * vec_w[0] + max_h * vec_h[0]),
            float(min_w * vec_w[1] + max_h * vec_h[1]),
        ],
    ]
    return {
        "major": float(max(w_tight, h_tight)),
        "minor": float(min(w_tight, h_tight)),
        "x": float(cx),
        "y": float(cy),
        "angle": float(angle),
        "w": float(w_tight),
        "h": float(h_tight),
        "bbox": bbox,
    }
