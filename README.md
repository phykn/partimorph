# PartiMorph

PartiMorph is a binary-mask-based particle morphology analysis library. It provides a consistent API for the following four metrics:

- Wadell Roundness
- ISO Circularity
- Riley Sphericity
- Aspect Ratio

## Highlights

- Single entry point: `analyze_mask(mask)`
- Built-in preprocessing: Largest Connected Component + hole filling
- Synthetic particle generation: `create_fourier_particle_mask(...)`
- Visualization utility: `plot_analysis_results(...)`

## Coordinate Convention

- Image/mask coordinates: `(y, x)`
- Geometric point arrays (contours, vertices): `(x, y)`

## Installation

Requires Python `3.12+`.

```bash
pip install -e .
```

## Quick Start

```python
import partimorph as pm

mask, info = pm.utils.create_fourier_particle_mask(
    shape=(512, 512),
    center=(256, 256),  # (y, x)
    sphericity=0.75,
    roundness=0.80,
    base_radius=100,
    return_info=True,
)

results = pm.analyze_mask(mask)

print("Roundness:", results["roundness"]["val"])
print("Circularity:", results["circularity"]["val"])
print("Sphericity:", results["sphericity"]["val"])
print("Aspect Ratio:", results["aspect_ratio"]["val"])
print("Target met:", info["target_met"])

pm.utils.plot_analysis_results(mask, results, title="PartiMorph Analysis")
```

## `analyze_mask()`

```python
analyze_mask(
    mask,
    *,
    use_aspect_ratio=True,
    use_roundness=True,
    use_circularity=True,
    use_sphericity=True,
    roundness_params=None,
    eps=0.001,
)
```

### Input Contract (Strict)

- `mask` must be a 2D `numpy.ndarray`.
- Allowed values are only `bool` or `{0, 1}`.
- If a float mask contains `NaN`/`inf`, a `ValueError` is raised.
- Internally, the mask is normalized to `uint8` binary (`0/1`).

### Return Rules

- Returns `None` for an empty mask
- Returns a result `dict` otherwise
- Metrics with `use_* = False` are omitted from the result keys

## Result Schema

```python
{
  "roundness": {"val": float} | None,
  "circularity": {"val": float} | None,
  "sphericity": {
    "val": float,
    "inscribed": {"x": int, "y": int, "r": float},
    "enclosing": {"x": int, "y": int, "r": float},
  } | None,
  "aspect_ratio": {
    "val": float,
    "ellipse": {
      "major": float,
      "minor": float,
      "x": float,
      "y": float,
      "angle": float,
      "w": float,
      "h": float,
      "bbox": list[list[float]],
    },
  } | None,
}
```

## Fourier Synthetic Mask Generation

```python
pm.utils.create_fourier_particle_mask(
    shape,
    center,
    sphericity,
    roundness,
    base_radius,
    frequencies=None,
    decay=1.0,
    num_angles=256,
    max_iter=20,
    amp_max=0.45,
    metric_tol=0.001,
    report_tol=0.1,
    seed=None,
    return_info=False,
)
```

Parameter notes:
- `metric_tol`: Internal search precision
- `report_tol`: Tolerance used only to compute `target_met` in the report

## Metric Definitions

- Roundness: Wadell roundness based on corner curvature
- Circularity: `4πA / P²`
- Sphericity: `R_inscribed / R_enclosing`
- Aspect Ratio: `major / minor` (always `>= 1`)

## References

- Wadell, H. (1932). *Volume, shape, and roundness of rock particles*.
- Riley, N. A. (1941). *Projection sphericity*.
- ISO 9276-6. *Quantitative representation of morphology*.
