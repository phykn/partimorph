import numpy as np
from scipy import ndimage
from typing import TypedDict
from .metrics import (
    AspectRatioResult,
    CircularityResult,
    RoundnessResult,
    SphericityResult,
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)


class AnalysisResult(TypedDict, total=False):
    roundness: RoundnessResult | None
    circularity: CircularityResult | None
    sphericity: SphericityResult | None
    aspect_ratio: AspectRatioResult | None


def analyze_mask(
    mask: np.ndarray,
    *,
    use_aspect_ratio: bool = True,
    use_roundness: bool = True,
    use_circularity: bool = True,
    use_sphericity: bool = True,
    roundness_params: dict[str, float] | None = None,
    eps: float = 0.001,
) -> AnalysisResult | None:
    """Analyze morphology metrics from a binary mask.

    Preprocessing is applied before metric computation:
    1) keep only the largest connected component (LCC) to remove small noise
    2) fill internal holes to stabilize downstream geometric metrics
    """
    if not np.any(mask):
        return None

    labeled, num_features = ndimage.label(mask)

    if num_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        mask = labeled == np.argmax(sizes)

    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    results: AnalysisResult = {}

    if use_aspect_ratio:
        results["aspect_ratio"] = compute_aspect_ratio(mask, eps=eps)

    if use_roundness:
        r_params = roundness_params or {}
        results["roundness"] = compute_roundness(mask, **r_params)

    if use_circularity:
        results["circularity"] = compute_circularity(mask, eps=eps)

    if use_sphericity:
        results["sphericity"] = compute_sphericity(mask, eps=eps)

    return results
