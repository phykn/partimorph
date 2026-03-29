import cv2
import numpy as np


def get_contours(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    cropped_mask, offset_x, offset_y = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return ()

    if cropped_mask.dtype != np.uint8:
        cropped_mask = cropped_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    offset = np.array([[[offset_x, offset_y]]], dtype=np.int32)
    return tuple(contour + offset for contour in contours)


def crop_mask(mask: np.ndarray, pad: int = 1) -> tuple[np.ndarray, int, int]:
    if mask.dtype != np.uint8:
        binary_mask_uint8 = mask.astype(np.uint8)
    else:
        binary_mask_uint8 = mask

    x_min, y_min, width, height = cv2.boundingRect(binary_mask_uint8)

    if width == 0 or height == 0:
        return (np.zeros((0, 0), dtype=mask.dtype), 0, 0)

    cropped = mask[y_min : y_min + height, x_min : x_min + width]

    if pad > 0:
        cropped = np.pad(cropped, pad_width=pad, mode="constant", constant_values=0)

    offset_y = y_min - pad
    offset_x = x_min - pad

    return (cropped, offset_x, offset_y)
