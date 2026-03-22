import numpy as np


def validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Validate and normalize a strict 2D binary mask.

    Returns a uint8 binary mask with values in {0, 1}.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray.")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D.")

    if np.issubdtype(mask.dtype, np.floating):
        if not np.isfinite(mask).all():
            raise ValueError("mask must not contain NaN or inf.")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    unique_vals = np.unique(mask)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError("mask must be binary with values in {0, 1}.")

    return mask.astype(np.uint8)
