import numpy as np
from scipy import ndimage
from typing import TypedDict

from .fitting import CircleData
from .metrics import (
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)


class AnalysisResult(TypedDict, total=False):
    roundness: dict | None
    circularity: dict | None
    sphericity: dict | None
    aspect_ratio: dict | None


def analyze_mask(
    mask: np.ndarray,
    *,
    use_ellipse: bool = True,
    use_roundness: bool = True,
    use_circularity: bool = True,
    use_sphericity: bool = True,
    roundness_params: dict[str, float] | None = None,
    eps: float = 1e-3,
) -> AnalysisResult | None:
    if not np.any(mask):
        return None

    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    padded_mask = np.pad(
        mask,
        pad_width = 1,
        mode = "constant",
        constant_values = 0,
    )

    results: AnalysisResult = {}

    if use_ellipse:
        res = compute_aspect_ratio(padded_mask, eps=eps)
        if res is not None:
            if res.get("ellipse"):
                res["ellipse"]["x"] -= 1
                res["ellipse"]["y"] -= 1
                if res["ellipse"].get("bbox"):
                    res["ellipse"]["bbox"] = [[x - 1, y - 1] for x, y in res["ellipse"]["bbox"]]
        results["aspect_ratio"] = res

    if use_roundness:
        r_params = roundness_params or {}
        results["roundness"] = compute_roundness(padded_mask, **r_params)

    if use_circularity:
        results["circularity"] = compute_circularity(padded_mask, eps=eps)

    if use_sphericity:
        res = compute_sphericity(padded_mask, eps=eps)
        if res is not None:
            res["inscribed"]["x"] -= 1
            res["inscribed"]["y"] -= 1
            res["enclosing"]["x"] -= 1
            res["enclosing"]["y"] -= 1
        results["sphericity"] = res

    return results