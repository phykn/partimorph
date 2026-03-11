import cv2
import numpy as np


def get_contours(mask: np.ndarray) -> tuple[np.ndarray, ...]:

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def crop_mask(
    mask: np.ndarray,
    pad: int = 1,
) -> tuple[np.ndarray, int, int]:

    mask_uint8 = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask_uint8)

    if w == 0 or h == 0:
        return np.zeros((0, 0), dtype=mask.dtype), 0, 0

    pad_y0 = max(0, y - pad)
    pad_y1 = min(mask.shape[0], y + h + pad)
    pad_x0 = max(0, x - pad)
    pad_x1 = min(mask.shape[1], x + w + pad)

    cropped = mask[pad_y0:pad_y1, pad_x0:pad_x1]

    return cropped, pad_x0, pad_y0
