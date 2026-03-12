# PartiMorph

A high-performance Python library for comprehensive particle shape analysis. It allows you to quantify and visualize multiple geometric metrics (Roundness, Circularity, Sphericity, Aspect Ratio) simultaneously from a single mask.

## Core Features

- **Multi-Metric Analysis**: Extract multiple geometric indicators at once.
  - **Roundness (Wadell)**: Classical roundness analysis based on corner curvature.
  - **Circularity**: Compactness measurement using area-to-perimeter ratio.
  - **Sphericity**: Morphology analysis via inscribed and enclosing circle ratios.
  - **Ellipse Fitting**: Extraction of major/minor axes and aspect ratios.
- **Mask Generation**: Built-in utilities to create masks for standard shapes (Circle, Ellipse, Star, etc.).
- **Parametric Masks**: Generate particle masks from sphericity and roundness targets using Fourier perturbations.
- **Visualization**: Automatic overlay plotting for inscribed/enclosing circles and bounding boxes.

## Quick Start

```python
import partimorph as pm

# 1. Create a sample mask (e.g., Star shape)
mask = pm.utils.create_star_mask(shape=(400, 400), center=(200, 200), outer_radius=100, inner_radius=40)

# 2. Execute comprehensive shape analysis
results = pm.analyze_mask(mask)

# 3. Access multiple metrics from the results
print(f"Roundness: {results['roundness']['val']:.2f}")
print(f"Circularity: {results['circularity']['val']:.2f}")
print(f"Sphericity: {results['sphericity']['val']:.2f}")

# 4. Visualize the results
pm.utils.plot_analysis_results(mask, results)
```

## Installation

Requires **Python 3.12+**.

```bash
# Install in editable mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Metrics Detail

| Metric | Description | Reference |
| :--- | :--- | :--- |
| **Roundness** | Average curvature of surface corners | Wadell (1932) |
| **Circularity** | $4\pi \times Area / Perimeter^2$ | ISO 9276-6 |
| **Sphericity** | $R_{inscribed} / R_{enclosing}$ | Riley (1941) |
| **Aspect Ratio** | Minor axis / Major axis | - |

## References

- [wadell_rs](https://github.com/PaPieta/wadell_rs): GitHub reference for Wadell Roundness algorithm implementation.
