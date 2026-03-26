import cv2
import numpy as np
from scipy import ndimage
from .schema import (
    AnalysisResult,
    AspectRatioResult,
    CircularityResult,
    RoundnessResult,
    SphericityResult,
)
from .metrics import (
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)
from .validation import to_binary


def analyze_mask(
    mask: np.ndarray,
    *,
    use_aspect_ratio: bool = True,
    use_roundness: bool = True,
    use_circularity: bool = True,
    use_sphericity: bool = True,
    roundness_params: dict[str, float] | None = None,
    eps: float = 0.001,
    target_dim: int = 384,
) -> AnalysisResult | None:

    mask_bool = to_binary(mask)
    if not np.any(mask_bool):
        return None

    original_height, original_width = mask_bool.shape[:2]
    max_dimension = max(original_height, original_width)

    is_resized = False
    if max_dimension > target_dim:
        scale = target_dim / max_dimension
        processed_mask = cv2.resize(
            mask_bool.astype(np.uint8),
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        mask_bool = processed_mask.astype(bool)
        is_resized = True

    labeled_mask, num_features = ndimage.label(mask_bool)

    if num_features > 1:
        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0
        mask_bool = labeled_mask == np.argmax(sizes)

    final_mask = ndimage.binary_fill_holes(mask_bool).astype(np.uint8)

    results: AnalysisResult = {}

    if use_aspect_ratio:
        results["aspect_ratio"] = compute_aspect_ratio(final_mask, eps=eps)

    if use_roundness:
        roundness_parameters = roundness_params or {}
        results["roundness"] = compute_roundness(final_mask, **roundness_parameters)

    if use_circularity:
        results["circularity"] = compute_circularity(final_mask, eps=eps)

    if use_sphericity:
        results["sphericity"] = compute_sphericity(final_mask, eps=eps)

    if is_resized and results:
        inverse_fx = original_width / final_mask.shape[1]
        inverse_fy = original_height / final_mask.shape[0]
        average_inverse_scale = (inverse_fx + inverse_fy) / 2.0

        if results.get("sphericity"):
            for key in ["inscribed", "enclosing"]:
                c = results["sphericity"][key]
                c["x"] = float(c["x"] * inverse_fx)
                c["y"] = float(c["y"] * inverse_fy)
                c["r"] = float(c["r"] * average_inverse_scale)

        if results.get("aspect_ratio"):
            ellipse = results["aspect_ratio"]["ellipse"]
            ellipse["x"] = float(ellipse["x"] * inverse_fx)
            ellipse["y"] = float(ellipse["y"] * inverse_fy)
            ellipse["major"] = float(ellipse["major"] * average_inverse_scale)
            ellipse["minor"] = float(ellipse["minor"] * average_inverse_scale)
            ellipse["w"] = float(ellipse["w"] * inverse_fx)
            ellipse["h"] = float(ellipse["h"] * inverse_fy)
            if "bbox" in ellipse:
                ellipse["bbox"] = [
                    [float(pt[0] * inverse_fx), float(pt[1] * inverse_fy)]
                    for pt in ellipse["bbox"]
                ]

    return results
