import cv2
import numpy as np


def _create_poly_mask(
    shape: tuple[int, int],
    vertices: np.ndarray,
) -> np.ndarray:
    res = np.zeros(shape, dtype=np.uint8)
    pts = vertices.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(res, [pts], 1)

    return res.astype(bool)


def _ellipse_radius(a: float, b: float, theta: np.ndarray) -> np.ndarray:
    # Exact polar form of ellipse: r(theta) = (a*b) / sqrt((b cos)^2 + (a sin)^2)
    denom = np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    denom = np.maximum(denom, 1e-12)

    return (a * b) / denom


def _roughness_signal(
    theta: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    decay: float,
) -> np.ndarray:
    weights = 1.0 / np.maximum(frequencies.astype(float), 1.0) ** decay
    signal = np.zeros_like(theta, dtype=float)

    for w, n, p in zip(weights, frequencies, phases, strict=True):
        signal += w * np.cos(n * theta + p)

    max_abs = np.max(np.abs(signal))
    if max_abs < 1e-12:
        return signal

    return signal / max_abs


def _segments_intersect(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> bool:
    """Return True if segments a1-a2 and b1-b2 intersect (proper or collinear)."""

    def _orient(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def _on_segment(p, q, r) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    if o1 == 0.0 and _on_segment(a1, b1, a2):
        return True
    if o2 == 0.0 and _on_segment(a1, b2, a2):
        return True
    if o3 == 0.0 and _on_segment(b1, a1, b2):
        return True
    if o4 == 0.0 and _on_segment(b1, a2, b2):
        return True

    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _is_simple_polygon(vertices: np.ndarray) -> bool:
    """Check if polygon is simple (no self intersections)."""

    n = len(vertices)
    if n < 4:
        return True

    for i in range(n):
        a1 = vertices[i]
        a2 = vertices[(i + 1) % n]
        for j in range(i + 1, n):
            if j in {i, (i + 1) % n, (i - 1) % n}:
                continue
            if (j + 1) % n == i:
                continue
            b1 = vertices[j]
            b2 = vertices[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return False

    return True


def _measure_roundness(mask: np.ndarray, metric: str) -> float:
    if metric == "circularity":
        from ..metrics import compute_circularity

        return float(compute_circularity(mask))
    if metric == "wadell":
        from ..metrics import compute_roundness

        return float(compute_roundness(mask))

    raise ValueError(f"Unsupported roundness metric: {metric}")


def create_fourier_particle_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    sphericity: float,
    roundness: float,
    base_radius: float,
    frequencies: list[int] | np.ndarray | None = None,
    decay: float = 1.0,
    num_angles: int = 512,
    metric: str = "wadell",
    max_iter: int = 20,
    amp_max: float = 0.45,
    metric_tol: float = 1e-3,
    ensure_simple: bool = True,
    seed: int | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Create a particle mask from sphericity and roundness.

    Args:
        shape: (H, W) mask shape.
        center: (cy, cx) center in pixel coordinates.
        sphericity: Aspect ratio proxy in (0, 1]. S=1 is circular.
        roundness: Target roundness in (0, 1]. Metric depends on metric.
        base_radius: Semi-major axis length in pixels.
        frequencies: Fourier frequencies (n >= 2 recommended).
        decay: Amplitude decay with frequency (1/n^decay).
        num_angles: Number of angular samples.
        metric: "wadell" or "circularity".
        max_iter: Max binary search iterations.
        amp_max: Maximum roughness amplitude.
        metric_tol: Early stop tolerance for metric match.
        ensure_simple: If True, prevents self-intersecting polygons.
        seed: RNG seed for reproducible phases.
        return_info: If True, returns (mask, info).
    """

    sphericity = float(np.clip(sphericity, 1e-3, 1.0))
    roundness = float(np.clip(roundness, 0.0, 1.0))

    if frequencies is None:
        frequencies = np.arange(2, 9)
    frequencies = np.array(frequencies, dtype=int)
    if np.any(frequencies < 2):
        raise ValueError("Frequencies should be >= 2 to avoid center drift.")

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(frequencies))

    theta = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    a = float(base_radius)
    b = float(base_radius) * sphericity

    r0 = _ellipse_radius(a, b, theta)
    rough = _roughness_signal(theta, frequencies, phases, decay=decay)

    cy, cx = center

    def build_vertices(amp: float) -> np.ndarray | None:
        r = r0 * (1.0 + amp * rough)
        r = np.maximum(r, 0.5)
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        vertices = np.stack([xs, ys], axis=1)
        if ensure_simple and not _is_simple_polygon(vertices):
            return None
        return vertices

    def build_mask(amp: float) -> np.ndarray | None:
        vertices = build_vertices(amp)
        if vertices is None:
            return None
        return _create_poly_mask(shape, vertices)

    # Baseline metric at amp=0 (smooth ellipse)
    mask0 = build_mask(0.0)
    if mask0 is None:
        raise RuntimeError("Failed to build a simple baseline polygon.")
    metric0 = _measure_roundness(mask0, metric)
    if metric0 <= roundness + metric_tol:
        info = {
            "sphericity_target": sphericity,
            "roundness_target": roundness,
            "roundness_metric": metric,
            "roundness_achieved": metric0,
            "amplitude": 0.0,
            "frequencies": frequencies,
            "phases": phases,
        }
        return (mask0, info) if return_info else mask0

    # Binary search amplitude to match target roundness
    low, high = 0.0, amp_max
    best_amp = 0.0
    best_val = metric0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mask_mid = build_mask(mid)
        if mask_mid is None:
            high = mid
            continue
        val = _measure_roundness(mask_mid, metric)

        if abs(val - roundness) < abs(best_val - roundness):
            best_amp, best_val = mid, val

        if abs(val - roundness) <= metric_tol:
            best_amp, best_val = mid, val
            break

        if val > roundness:
            low = mid
        else:
            high = mid

    mask_best = build_mask(best_amp)
    if mask_best is None:
        mask_best = mask0
        best_amp = 0.0
        best_val = metric0
    info = {
        "sphericity_target": sphericity,
        "roundness_target": roundness,
        "roundness_metric": metric,
        "roundness_achieved": best_val,
        "amplitude": best_amp,
        "frequencies": frequencies,
        "phases": phases,
    }

    return (mask_best, info) if return_info else mask_best
