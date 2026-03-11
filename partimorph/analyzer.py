import numpy as np
from scipy import ndimage

from .metrics import (
    compute_circularity,
    compute_roundness,
    find_enclosing_circle,
    find_inscribed_circle,
    fit_ellipse,
)


def analyze_mask(
    mask: np.ndarray,
    use_ellipse: bool = True,
    use_roundness: bool = True,
    use_circularity: bool = True,
    use_sphericity: bool = True,
) -> dict:

    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)

    res = {}

    if use_ellipse:
        ellipse_data = fit_ellipse(padded_mask)
        if ellipse_data is not None:
            ellipse_data["bbox"] -= 1.0
            res["ellipse"] = {
                "val": ellipse_data["ratio"],
                "major": ellipse_data["major"],
                "minor": ellipse_data["minor"],
                "bbox": ellipse_data["bbox"],
            }
        else:
            res["ellipse"] = None

    if use_roundness:
        res["roundness"] = {"val": np.clip(compute_roundness(padded_mask), 0.0, 1.0)}

    if use_circularity:
        res["circularity"] = {"val": np.clip(compute_circularity(padded_mask), 0.0, 1.0)}

    if use_sphericity:
        cx_in, cy_in, r_in = find_inscribed_circle(padded_mask)
        cx_en, cy_en, r_en = find_enclosing_circle(padded_mask)

        sphericity_val = np.clip(r_in / r_en, 0.0, 1.0) if r_en > 0.0 else 0.0

        res["sphericity"] = {
            "val": sphericity_val,
            "inscribed": {
                "x": cx_in - 1,
                "y": cy_in - 1,
                "r": r_in,
            },
            "enclosing": {
                "x": cx_en - 1,
                "y": cy_en - 1,
                "r": r_en,
            },
        }

    return res
