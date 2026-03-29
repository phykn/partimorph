import numpy as np
from ..metrics import compute_roundness
from .geometry import create_poly_mask, polar_vertices


def _build_info(
    *,
    sphericity: float,
    roundness: float,
    achieved: float,
    report_tol: float,
    amplitude: float,
    frequencies: np.ndarray,
    phases: np.ndarray,
) -> dict:
    abs_error = abs(roundness - achieved)
    return {
        "sphericity_target": sphericity,
        "roundness_target": roundness,
        "roundness_achieved": achieved,
        "abs_error": abs_error,
        "report_tol": report_tol,
        "target_met": abs_error <= report_tol,
        "amplitude": amplitude,
        "frequencies": frequencies.tolist(),
        "phases": phases.tolist(),
    }


def ellipse_radius(
    a: float, b: float, cos_t: np.ndarray, sin_t: np.ndarray
) -> np.ndarray:
    denom = np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)
    return a * b / np.maximum(denom, 1e-12)


def roughness_signal(
    thetas: np.ndarray, frequencies: np.ndarray, phases: np.ndarray, decay: float
) -> np.ndarray:
    weights = 1.0 / np.maximum(frequencies.astype(float), 1.0) ** decay
    components = weights[:, None] * np.cos(
        frequencies[:, None] * thetas[None, :] + phases[:, None]
    )
    signal = np.sum(components, axis=0)
    max_abs = np.max(np.abs(signal))
    return signal / max_abs if max_abs > 1e-12 else signal


def create_particle_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    sphericity: float,
    roundness: float,
    *,
    num_angles: int = 256,
    frequencies: list[int] | np.ndarray | None = None,
    decay: float = 1.0,
    seed: int | None = None,
    max_iter: int = 20,
    amp_max: float = 0.45,
    metric_tol: float = 0.001,
    report_tol: float = 0.1,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generates a synthetic particle mask with target morphological features.

    Uses a Fourier-based approach to generate a closed polygon-based mask that
    matches specified sphericity and roundness values through iterative optimization
    of the roughness amplitude.

    Args:
        shape: Dimensions of the output mask (height, width).
        center: Center coordinates of the particle (y, x).
        radius: Base radius of the particle.
        sphericity: Target sphericity (0 to 1).
        roundness: Target roundness (0 to 1).
        num_angles: Number of vertices in the generated polygon.
        frequencies: List of frequencies used to generate surface roughness.
        decay: Frequency decay rate of the roughness signal.
        seed: Random seed for phase generation.
        max_iter: Max iterations for roundness calibration.
        amp_max: Maximum allowed amplitude for the roughness signal.
        metric_tol: Internal precision for reaching the target roundness.
        report_tol: Error threshold for the 'target_met' report flag.
        return_info: If True, returns a metadata dictionary along with the mask.

    Returns:
        The binary mask as a numpy array, or a tuple of (mask, info_dict).
    """
    sphericity = np.clip(sphericity, 0.001, 1.0)
    roundness = np.clip(roundness, 0.0, 1.0)
    radius = float(radius)

    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Invalid radius")
    if num_angles < 3:
        raise ValueError("Invalid angles")

    frequencies_arr = np.asarray(
        frequencies if frequencies is not None else np.arange(2, 9), dtype=int
    )
    phases = np.random.default_rng(seed).uniform(
        0.0, 2.0 * np.pi, size=len(frequencies_arr)
    )
    cy, cx = center

    if radius == 0.0:
        mask = np.zeros(shape, dtype=bool)
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            mask[iy, ix] = True

        info = _build_info(
            sphericity=sphericity,
            roundness=roundness,
            achieved=0.0,
            report_tol=report_tol,
            amplitude=0.0,
            frequencies=frequencies_arr,
            phases=phases,
        )
        return (mask, info) if return_info else mask

    thetas_sim = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    base_sim = ellipse_radius(
        radius, radius * sphericity, np.cos(thetas_sim), np.sin(thetas_sim)
    )
    noise_sim = roughness_signal(thetas_sim, frequencies_arr, phases, decay) * radius
    scale = min(1.0, 128.0 / radius)

    size_sim = int(np.ceil((radius * scale * (1.0 + amp_max) + 2) * 2))
    form_sim = (size_sim, size_sim)
    ct_sim = size_sim / 2.0

    def measure(amp_val: float) -> float:
        rs = (base_sim + amp_val * noise_sim) * scale
        vs = polar_vertices((ct_sim, ct_sim), np.maximum(rs, 0.5), thetas_sim)
        res = compute_roundness(create_poly_mask(form_sim, vs))
        return float(res["val"]) if res else 0.0

    val = measure(0.0)
    amp = 0.0
    if val > roundness + metric_tol:
        low, high = 0.0, float(amp_max)
        for _ in range(max_iter):
            mid = (low + high) / 2
            val_mid = measure(mid)
            if abs(val_mid - roundness) < abs(val - roundness):
                val, amp = val_mid, mid
            if abs(val_mid - roundness) <= metric_tol:
                break
            if val_mid > roundness:
                low = mid
            else:
                high = mid

    thetas = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    rs_base = ellipse_radius(
        radius, radius * sphericity, np.cos(thetas), np.sin(thetas)
    )
    rs_noise = roughness_signal(thetas, frequencies_arr, phases, decay) * radius
    rs_final = np.maximum(rs_base + amp * rs_noise, 0.5)

    vs = polar_vertices((cy, cx), rs_final, thetas)
    mask = create_poly_mask(shape, vs).astype(bool)

    if not return_info:
        return mask

    info = _build_info(
        sphericity=sphericity,
        roundness=roundness,
        achieved=val,
        report_tol=report_tol,
        amplitude=amp,
        frequencies=frequencies_arr,
        phases=phases,
    )
    return mask, info
