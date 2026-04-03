import cv2
import numpy as np
from .schema import CircleData, EllipseData, Mask, Points
from .misc import crop_mask, get_contours


def _ellipse_payload(
    *,
    center_x: float,
    center_y: float,
    angle: float,
    width: float,
    height: float,
    bbox: list[list[float]],
) -> EllipseData:
    return {
        "major": float(max(width, height)),
        "minor": float(min(width, height)),
        "x": float(center_x),
        "y": float(center_y),
        "angle": float(angle),
        "w": float(width),
        "h": float(height),
        "bbox": bbox,
    }


def find_inscribed_circle(mask: Mask) -> CircleData | None:
    cropped_mask, pad_x0, pad_y0 = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return None

    distance_transform = cv2.distanceTransform(
        cropped_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    max_idx = np.argmax(distance_transform)
    crop_center_y, crop_center_x = np.unravel_index(max_idx, distance_transform.shape)

    center_x = float(int(crop_center_x) + pad_x0)
    center_y = float(int(crop_center_y) + pad_y0)
    radius = float(distance_transform[crop_center_y, crop_center_x])

    return {"x": center_x, "y": center_y, "r": radius}


def find_enclosing_circle(mask: Mask) -> CircleData | None:
    if not np.any(mask):
        return None

    contours = get_contours(mask)
    if not contours:
        return None

    points = np.concatenate(contours)
    hull = cv2.convexHull(points)

    (center_x, center_y), radius = cv2.minEnclosingCircle(hull)

    x = float(center_x)
    y = float(center_y)
    r = float(radius)

    return {"x": x, "y": y, "r": r}


def fit_ellipse(mask: Mask) -> EllipseData | None:
    contours = get_contours(mask)
    if not contours:
        return None

    points: Points = np.concatenate(contours).astype(np.float32)
    if len(points) < 5:
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (width, height), angle = rect

        if width <= 1e-06 or height <= 1e-06:
            return None

        box = cv2.boxPoints(rect)
        bbox = [[float(x), float(y)] for x, y in box]
        return _ellipse_payload(
            center_x=center_x,
            center_y=center_y,
            angle=angle,
            width=width,
            height=height,
            bbox=bbox,
        )

    _, _, angle = cv2.fitEllipse(points)
    radians = np.deg2rad(angle)
    cos_a, sin_a = (np.cos(radians), np.sin(radians))

    width_vector = np.array([cos_a, sin_a])
    height_vector = np.array([-sin_a, cos_a])

    width_projection = points @ width_vector
    height_projection = points @ height_vector

    min_width, max_width = (width_projection.min(), width_projection.max())
    min_height, max_height = (height_projection.min(), height_projection.max())

    tight_width = max_width - min_width
    tight_height = max_height - min_height

    mid_width = (max_width + min_width) / 2.0
    mid_height = (max_height + min_height) / 2.0

    center_x = mid_width * width_vector[0] + mid_height * height_vector[0]
    center_y = mid_width * width_vector[1] + mid_height * height_vector[1]

    bbox = [
        [
            float(min_width * width_vector[0] + min_height * height_vector[0]),
            float(min_width * width_vector[1] + min_height * height_vector[1]),
        ],
        [
            float(max_width * width_vector[0] + min_height * height_vector[0]),
            float(max_width * width_vector[1] + min_height * height_vector[1]),
        ],
        [
            float(max_width * width_vector[0] + max_height * height_vector[0]),
            float(max_width * width_vector[1] + max_height * height_vector[1]),
        ],
        [
            float(min_width * width_vector[0] + max_height * height_vector[0]),
            float(min_width * width_vector[1] + max_height * height_vector[1]),
        ],
    ]

    return _ellipse_payload(
        center_x=center_x,
        center_y=center_y,
        angle=angle,
        width=tight_width,
        height=tight_height,
        bbox=bbox,
    )
