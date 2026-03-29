# PartiMorph

PartiMorph analyzes a 2D binary particle mask and returns:
- Wadell roundness
- ISO circularity
- Riley sphericity
- Aspect ratio

## Install

Python `3.12+`

```bash
pip install -e .
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

## Input Rules

- `mask` must be a 2D `numpy.ndarray`
- Allowed values: `bool` or `{0, 1}`
- Empty mask returns `None`

## Main API

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

Notes:
- Large masks are downscaled automatically (`target_dim`) for speed.
- `use_* = False` removes that metric from the output keys.

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

## Utilities

- `pm.utils.create_particle_mask(...)`: synthetic mask generator
- `pm.utils.plot_analysis_results(mask, results)`: quick visualization
