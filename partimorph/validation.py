import numpy as np


def to_binary(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray.")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D.")

    if np.issubdtype(mask.dtype, np.floating):
        if not np.isfinite(mask).all():
            raise ValueError("mask must not contain NaN or inf.")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    unique_values = np.unique(mask)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError("mask must be binary with values in {0, 1}.")

    return mask.astype(np.uint8)
