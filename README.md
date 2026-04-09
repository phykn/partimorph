# PartiMorph

2D binary particle shape analysis library. Computes four morphological metrics from a binary mask:

- **Wadell roundness** — corner curvature-based roundness
- **ISO circularity** — perimeter-area ratio
- **Riley sphericity** — inscribed/enclosing circle ratio
- **Aspect ratio** — fitted ellipse major/minor axis ratio

All values are dimensionless ratios (0–1, except aspect ratio which is >= 1).

## Installation

Python `3.12+`

```bash
pip install partimorph
```

## Quick Start

```python
import partimorph as pm

mask = pm.utils.create_circle_mask((256, 256), (128, 128), 60)
results = pm.analyze_mask(mask)

print(results["roundness"]["val"])
print(results["circularity"]["val"])
print(results["sphericity"]["val"])
print(results["aspect_ratio"]["val"])
```

Visualize the results:

```python
pm.utils.plot_analysis_results(mask, results)
```

## API

```python
pm.analyze_mask(
    mask,
    use_aspect_ratio=True,
    use_roundness=True,
    use_circularity=True,
    use_sphericity=True,
    roundness_params=None,
    eps=0.001,
    target_dim=384,
)
```

| Parameter | Description |
|---|---|
| `mask` | 2D `numpy.ndarray` with `bool` or `{0, 1}` values |
| `use_*` | Toggle individual metrics on/off |
| `roundness_params` | Optional dict to override Wadell roundness parameters |
| `eps` | Tolerance for geometric computations |
| `target_dim` | Large masks are downscaled to this size for speed |

Returns an `AnalysisResult` dict. Keys are only present for enabled metrics (`use_*=True`).

## Result Shape

```python
{
  "roundness": {"val": float} | None,
  "circularity": {"val": float} | None,
  "sphericity": {
    "val": float,
    "inscribed": {"x": float, "y": float, "r": float},
    "enclosing": {"x": float, "y": float, "r": float},
  } | None,
  "aspect_ratio": {
    "val": float,
    "ellipse": {
      "major": float, "minor": float,
      "x": float, "y": float, "angle": float,
      "w": float, "h": float, "bbox": list[list[float]],
    },
  } | None,
}
```

## Input Rules

- `mask` must be a 2D `numpy.ndarray`
- Allowed values: `bool` or `{0, 1}`
- Empty mask returns `None`

## Utilities

- `pm.utils.create_particle_mask(...)` — synthetic mask generator with Fourier roughness control
- `pm.utils.plot_analysis_results(mask, results)` — overlay visualization of all computed metrics

## License

MIT
