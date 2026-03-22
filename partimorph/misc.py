import cv2
import numpy as np


def get_contours(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    cropped, pad_x0, pad_y0 = crop_mask(mask, pad=1)
    if cropped.size == 0:
        return ()

    cropped_uint8 = cropped if cropped.dtype == np.uint8 else cropped.astype(np.uint8)
    contours, _ = cv2.findContours(
        cropped_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    shifted_contours = []
    for cnt in contours:
        cnt_shifted = cnt + [pad_x0, pad_y0]
        shifted_contours.append(cnt_shifted)

    return tuple(shifted_contours)


def crop_mask(
    mask: np.ndarray,
    pad: int = 1,
) -> tuple[np.ndarray, int, int]:
    """Crop mask to its bounding box for performance optimization.

    This function isolates the active region (pixels=1) from the global mask,
    significantly reducing the computational grid size for distance transforms
    and contour detection algorithms.

    The optional `pad` argument (default=1) leaves a border of zeros around the
    cropped shape, which is critical to ensure that boundary pixels do not touch
    the image edges and prevent closed contour extraction.

    Returns:
        cropped_mask: The optimized local mask.
        pad_x0: The global X offset of the cropped region's top-left corner.
        pad_y0: The global Y offset of the cropped region's top-left corner.
    """

    mask_uint8 = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask_uint8)

    if w == 0 or h == 0:
        return np.zeros((0, 0), dtype=mask.dtype), 0, 0

    cropped = mask[y : y + h, x : x + w]

    if pad > 0:
        cropped = np.pad(
            cropped,
            pad_width=pad,
            mode="constant",
            constant_values=0,
        )

    pad_y0 = y - pad
    pad_x0 = x - pad

    return cropped, pad_x0, pad_y0
