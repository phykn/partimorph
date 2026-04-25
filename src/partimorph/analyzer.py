import cv2
import numpy as np
from scipy import ndimage
from .schema import AnalysisResult, Mask
from .metrics import (
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)
from .validation import to_binary


def _preprocess_mask(mask: Mask, target_dim: int) -> tuple[Mask, bool]:
    height, width = mask.shape[:2]
    max_dimension = max(height, width)
    is_resized = max_dimension > target_dim

    if is_resized:
        scale = target_dim / max_dimension
        mask = cv2.resize(
            mask.astype(np.uint8),
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    mask_bool = mask.astype(bool)

    labeled_mask, num_features = ndimage.label(mask_bool)
    if num_features > 1:
        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0
        mask_bool = labeled_mask == np.argmax(sizes)

    return (ndimage.binary_fill_holes(mask_bool).astype(np.uint8), is_resized)


def _rescale_results(
    results: AnalysisResult,
    *,
    scale_x: float,
    scale_y: float,
) -> None:
    mean_scale = (scale_x + scale_y) / 2.0

    if results.get("sphericity"):
        for key in ["inscribed", "enclosing"]:
            circle_data = results["sphericity"][key]
            circle_data["x"] = float(circle_data["x"] * scale_x)
            circle_data["y"] = float(circle_data["y"] * scale_y)
            circle_data["r"] = float(circle_data["r"] * mean_scale)

    if results.get("aspect_ratio"):
        ellipse = results["aspect_ratio"]["ellipse"]
        ellipse["x"] = float(ellipse["x"] * scale_x)
        ellipse["y"] = float(ellipse["y"] * scale_y)
        ellipse["major"] = float(ellipse["major"] * mean_scale)
        ellipse["minor"] = float(ellipse["minor"] * mean_scale)
        ellipse["w"] = float(ellipse["w"] * scale_x)
        ellipse["h"] = float(ellipse["h"] * scale_y)
        ellipse["bbox"] = [
            [float(pt[0] * scale_x), float(pt[1] * scale_y)] for pt in ellipse["bbox"]
        ]


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
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if target_dim <= 0:
        raise ValueError("target_dim must be > 0.")

    mask_binary = to_binary(mask)
    if not np.any(mask_binary):
        return None

    original_height, original_width = mask_binary.shape
    final_mask, is_resized = _preprocess_mask(mask_binary, target_dim=target_dim)

    results: AnalysisResult = {}
    if use_aspect_ratio:
        results["aspect_ratio"] = compute_aspect_ratio(final_mask, eps=eps)
    if use_roundness:
        results["roundness"] = compute_roundness(final_mask, **(roundness_params or {}))
    if use_circularity:
        results["circularity"] = compute_circularity(final_mask, eps=eps)
    if use_sphericity:
        results["sphericity"] = compute_sphericity(final_mask, eps=eps)

    if is_resized and results:
        scale_x = original_width / final_mask.shape[1]
        scale_y = original_height / final_mask.shape[0]
        _rescale_results(
            results,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    return results
