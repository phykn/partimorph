import numpy as np
from ..metrics import compute_roundness
from .geometry import create_poly_mask, polar_vertices


def _ellipse_radius(
    a: float, b: float, cos_t: np.ndarray, sin_t: np.ndarray
) -> np.ndarray:
    denom = np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)
    return a * b / np.maximum(denom, 1e-12)


def _roughness_signal(
    theta: np.ndarray, frequencies: np.ndarray, phases: np.ndarray, decay: float
) -> np.ndarray:
    weights = 1.0 / np.maximum(frequencies.astype(float), 1.0) ** decay
    components = weights[:, None] * np.cos(
        frequencies[:, None] * theta[None, :] + phases[:, None]
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
    max_iter: int = 20,
    amp_max: float = 0.45,
    metric_tol: float = 0.001,
    report_tol: float = 0.1,
    seed: int | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generate a Fourier particle mask.

    Args:
        shape: Output mask shape as (height, width).
        center: Particle center in image coordinates as (y, x).
        report_tol: Tolerance used only for reporting `target_met`.
    """
    sphericity = float(np.clip(sphericity, 0.001, 1.0))
    roundness = float(np.clip(roundness, 0.0, 1.0))

    if frequencies is None:
        frequencies = np.arange(2, 9)

    frequencies = np.array(frequencies, dtype=int)

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(frequencies))
    major_axis = float(base_radius)

    if not np.isfinite(major_axis):
        raise ValueError("base_radius must be a finite number.")
    if major_axis < 0.0:
        raise ValueError("base_radius must be >= 0.")
    if not isinstance(num_angles, (int, np.integer)):
        raise TypeError("num_angles must be an integer.")
    if num_angles < 3:
        raise ValueError("num_angles must be >= 3.")
    if not np.isfinite(report_tol):
        raise ValueError("report_tol must be a finite number.")
    if report_tol < 0.0:
        raise ValueError("report_tol must be >= 0.")

    cy, cx = center
    if major_axis == 0.0:
        final_mask = np.zeros(shape, dtype=bool)
        iy = int(round(cy))
        ix = int(round(cx))

        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            final_mask[iy, ix] = True

        if not return_info:
            return final_mask

        return (
            final_mask,
            {
                "sphericity_target": sphericity,
                "roundness_target": roundness,
                "roundness_achieved": 0.0,
                "abs_error": abs(roundness - 0.0),
                "report_tol": report_tol,
                "target_met": bool(abs(roundness - 0.0) <= report_tol),
                "amplitude": 0.0,
                "frequencies": frequencies.tolist(),
                "phases": phases.tolist(),
            },
        )

    num_angles_search = 128
    theta_search = np.linspace(0.0, 2.0 * np.pi, num_angles_search, endpoint=False)
    cos_search = np.cos(theta_search)
    sin_search = np.sin(theta_search)

    noise_search_raw = _roughness_signal(theta_search, frequencies, phases, decay)
    radius_search_raw = _ellipse_radius(
        a=major_axis, b=major_axis * sphericity, cos_t=cos_search, sin_t=sin_search
    )

    scale = 1.0 if major_axis == 0.0 else min(1.0, 128.0 / major_axis)
    base_r_search = radius_search_raw * scale
    noise_search = noise_search_raw * major_axis * scale

    local_size = int(np.ceil((major_axis * scale * (1.0 + amp_max) + 2) * 2))
    local_shape = (local_size, local_size)
    local_center = local_size / 2.0

    def measure_at_amp(amp: float) -> float:
        r = base_r_search + amp * noise_search
        r = np.maximum(r, 0.5)

        vertices = polar_vertices(
            center=(local_center, local_center), radii=r, angles=theta_search
        )

        mask_local = create_poly_mask(local_shape, vertices)

        res = compute_roundness(mask_local)

        return float(res["val"]) if res else 0.0

    val_current = measure_at_amp(0.0)
    target_amp = 0.0

    if val_current > roundness + metric_tol:
        low, high = (0.0, float(amp_max))

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
        a=major_axis, b=major_axis * sphericity, cos_t=cos_final, sin_t=sin_final
    )
    radius_final = np.maximum(
        radius_base_final + target_amp * major_axis * noise_final, 0.5
    )

    final_vertices = polar_vertices(center=(cy, cx), radii=radius_final, angles=theta_final)
    final_mask = create_poly_mask(shape, final_vertices).astype(bool)

    if not return_info:
        return final_mask

    abs_error = abs(roundness - val_current)

    return (
        final_mask,
        {
            "sphericity_target": sphericity,
            "roundness_target": roundness,
            "roundness_achieved": val_current,
            "abs_error": abs_error,
            "report_tol": report_tol,
            "target_met": bool(abs_error <= report_tol),
            "amplitude": target_amp,
            "frequencies": frequencies.tolist(),
            "phases": phases.tolist(),
        },
    )
