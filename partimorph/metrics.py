import numpy as np
import skimage.measure
from typing import TypedDict
from .fitting import (
    CircleData,
    EllipseData,
    find_enclosing_circle,
    find_inscribed_circle,
    fit_ellipse,
)
from .wadell import compute_roundness as _compute_roundness
from .misc import crop_mask


class RoundnessResult(TypedDict):
    val: float


class CircularityResult(TypedDict):
    val: float


class SphericityResult(TypedDict):
    val: float
    inscribed: CircleData
    enclosing: CircleData


class AspectRatioResult(TypedDict):
    val: float
    ellipse: EllipseData


def compute_roundness(
    mask: np.ndarray,
    *,
    max_dev_thresh: float = 0.3,
    circle_fit_thresh: float = 0.98,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> RoundnessResult | None:
    val = _compute_roundness(
        mask,
        max_dev_thresh=max_dev_thresh,
        circle_fit_thresh=circle_fit_thresh,
        alpha_ratio=alpha_ratio,
        beta_ratio=beta_ratio,
    )

    if val is None:
        return None

    val = float(np.clip(val, 0.0, 1.0))

    return {"val": val}


def compute_circularity(
    mask: np.ndarray, *, eps: float = 0.001
) -> CircularityResult | None:
    cropped_mask, _, _ = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return None

    perimeter = skimage.measure.perimeter_crofton(cropped_mask, 4)

    if perimeter < eps:
        return None

    area = float(np.count_nonzero(mask))
    val = 4.0 * np.pi * area / perimeter**2
    val = float(np.clip(val, 0.0, 1.0))

    return {"val": val}


def compute_sphericity(
    mask: np.ndarray, *, eps: float = 0.001
) -> SphericityResult | None:
    inscribed = find_inscribed_circle(mask)
    enclosing = find_enclosing_circle(mask)

    if inscribed is None or enclosing is None:
        return None

    r_in = inscribed["r"]
    r_en = enclosing["r"]

    if r_en < eps:
        return None

    val = float(np.clip(r_in / r_en, 0.0, 1.0))

    return {"val": val, "inscribed": inscribed, "enclosing": enclosing}


def compute_aspect_ratio(
    mask: np.ndarray, *, eps: float = 0.001
) -> AspectRatioResult | None:
    ellipse_data = fit_ellipse(mask)

    if ellipse_data is None:
        return None

    if ellipse_data["minor"] < eps:
        return None

    val = ellipse_data["major"] / ellipse_data["minor"]

    return {"val": val, "ellipse": ellipse_data}
