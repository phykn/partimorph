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

    # Keep only the largest connected component to avoid disjoint noise issues
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # Ignore background
        mask = labeled == np.argmax(sizes)

    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    results: AnalysisResult = {}

    if use_ellipse:
        results["aspect_ratio"] = compute_aspect_ratio(mask, eps=eps)

    if use_roundness:
        r_params = roundness_params or {}
        results["roundness"] = compute_roundness(mask, **r_params)

    if use_circularity:
        results["circularity"] = compute_circularity(mask, eps=eps)

    if use_sphericity:
        results["sphericity"] = compute_sphericity(mask, eps=eps)

    return results
