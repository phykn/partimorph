import cv2
import numpy as np

from ..metrics import compute_circularity, compute_roundness


def _create_poly_mask(
    shape: tuple[int, int],
    vertices: np.ndarray,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:

    res = np.zeros(shape, dtype=dtype)
    pts = vertices.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(res, [pts], 1)
    return res


def _ellipse_radius(
    a: float,
    b: float,
    cos_t: np.ndarray,
    sin_t: np.ndarray,
) -> np.ndarray:

    denom = np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)
    return (a * b) / np.maximum(denom, 1e-12)


def _roughness_signal(
    theta: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    decay: float,
) -> np.ndarray:

    weights = 1.0 / np.maximum(frequencies.astype(float), 1.0) ** decay

    components = weights[:, None] * np.cos(
        frequencies[:, None] * theta[None, :] + phases[:, None],
    )
    signal = np.sum(components, axis=0)
    max_abs = np.max(np.abs(signal))

    return signal / max_abs if max_abs > 1e-12 else signal


def create_fourier_particle_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    sphericity: float,
    roundness: float,
    base_radius: float,
    frequencies: list[int] | np.ndarray | None = None,
    decay: float = 1.0,
    num_angles: int = 256,
    metric: str = "wadell",
    max_iter: int = 20,
    amp_max: float = 0.45,
    metric_tol: float = 1e-3,
    seed: int | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    assert metric in ("wadell", "circularity"), (
        f"Invalid metric: '{metric}'. Supported metrics are ('wadell', 'circularity')."
    )

    sphericity = float(np.clip(sphericity, 1e-3, 1.0))
    roundness = float(np.clip(roundness, 0.0, 1.0))

    if frequencies is None:
        frequencies = np.arange(2, 9)
    frequencies = np.array(frequencies, dtype=int)

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(frequencies))
    major_axis = float(base_radius)

    num_angles_search = 128
    theta_search = np.linspace(0.0, 2.0 * np.pi, num_angles_search, endpoint=False)
    cos_search = np.cos(theta_search)
    sin_search = np.sin(theta_search)

    noise_search_raw = _roughness_signal(theta_search, frequencies, phases, decay)
    radius_search_raw = _ellipse_radius(
        a=major_axis,
        b=major_axis * sphericity,
        cos_t=cos_search,
        sin_t=sin_search,
    )

    # Scale down for invariant search time
    scale = min(1.0, 128.0 / major_axis)
    base_r_search = radius_search_raw * scale
    noise_search = noise_search_raw * major_axis * scale

    local_size = int(np.ceil((major_axis * scale * (1.0 + amp_max) + 2) * 2))
    local_shape = (local_size, local_size)
    local_center = local_size / 2.0

    def measure_at_amp(amp: float) -> float:
        r = base_r_search + (amp * noise_search)
        r = np.maximum(r, 0.5)

        vertices = np.stack(
            [local_center + r * cos_search, local_center + r * sin_search],
            axis=1,
        )

        mask_local = _create_poly_mask(local_shape, vertices)

        if metric == "circularity":
            res = compute_circularity(mask_local)
        else:
            res = compute_roundness(mask_local)

        return float(res["val"]) if res else 0.0

    val_current = measure_at_amp(0.0)
    target_amp = 0.0

    if val_current > roundness + metric_tol:
        low, high = 0.0, float(amp_max)
        for _ in range(max_iter):
            mid = (low + high) / 2
            val_mid = measure_at_amp(mid)

            if abs(val_mid - roundness) < abs(val_current - roundness):
                val_current = val_mid
                target_amp = mid

            if abs(val_mid - roundness) <= metric_tol:
                break

            if val_mid > roundness:
                low = mid
            else:
                high = mid

    theta_final = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    cos_final = np.cos(theta_final)
    sin_final = np.sin(theta_final)

    noise_final = _roughness_signal(theta_final, frequencies, phases, decay)
    radius_base_final = _ellipse_radius(
        a=major_axis,
        b=major_axis * sphericity,
        cos_t=cos_final,
        sin_t=sin_final,
    )

    radius_final = np.maximum(
        radius_base_final + (target_amp * major_axis * noise_final), 0.5
    )
    cx, cy = center

    final_vertices = np.stack(
        [cx + radius_final * cos_final, cy + radius_final * sin_final],
        axis=1,
    )

    final_mask = _create_poly_mask(shape, final_vertices).astype(bool)

    if not return_info:
        return final_mask

    return final_mask, {
        "sphericity_target": sphericity,
        "roundness_target": roundness,
        "roundness_metric": metric,
        "roundness_achieved": val_current,
        "amplitude": target_amp,
        "frequencies": frequencies.tolist(),
        "phases": phases.tolist(),
    }
