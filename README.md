# PartiMorph

A high-performance Python library for comprehensive and precise particle shape analysis. Quantify and visualize morphology (Roundness, Circularity, Sphericity, Aspect Ratio) from binary masks with scientific accuracy.

## Core Features

- **Multi-Metric Analysis**: Extract multiple geometric indicators simultaneously from a single mask.
  - **Roundness (Wadell)**: Classical corners-based curvature analysis using optimized algorithms.
  - **Circularity**: Compactness measurement based on the ISO 9276-6 standard.
  - **Sphericity**: Morphology analysis via inscribed/enclosing circle ratios (Riley's Sphericity).
  - **Ellipse Fitting & AR**: Precise extraction of major/minor axes using tight-fitting ellipse algorithms.
- **Advanced Parametric Generation**: 
  - **Fourier Particle Generator**: High-speed, high-precision particle generation matching target Sphericity and Roundness metrics.
  - **Adaptive Scaling**: High-performance binary search for targets, maintaining speed regardless of particle size.
  - **Geometric Utilities**: Create standard shapes like circles, ellipses, rectangles, triangles, and stars.
- **Scientific Visualization**: 
  - Automatic overlay of inscribed/enclosing circles.
  - Accurate fitted ellipses and **Oriented Bounding Boxes (OBB)**.

## Quick Start

```python
import partimorph as pm

# 1. Create a Fourier particle with specific targets
mask, info = pm.utils.create_fourier_particle_mask(
    shape = (512, 512),
    center = (256, 256),
    sphericity = 0.75,
    roundness = 0.8,
    base_radius = 100,
    return_info = True,
)

# 2. Execute comprehensive shape analysis
results = pm.analyze_mask(mask)

# 3. Access achieved results vs targets
print(f"Target Roundness: {info['roundness_target']} vs Achieved: {results['roundness']['val']:.4f}")
print(f"Sphericity (Riley): {results['sphericity']['val']:.4f}")
print(f"Aspect Ratio: {results['aspect_ratio']['val']:.4f}")

# 4. Visualize with fitted shapes and bounding boxes
pm.utils.plot_analysis_results(mask, results, title="Particle Morphology Analysis")
```

## Metrics Detail

| Metric           | Definition                           | Reference     |
| :--------------- | :----------------------------------- | :------------ |
| **Roundness**    | Average curvature of surface corners | Wadell (1932) |
| **Circularity**  | $4\pi \times Area / Perimeter^2$     | ISO 9276-6    |
| **Sphericity**   | $R_{inscribed} / R_{enclosing}$      | Riley (1941)  |
| **Aspect Ratio** | Major Axis / Minor Axis ($\ge 1.0$)  | -             |

## Installation

Requires **Python 3.12+**.

```bash
# Clone and install in editable mode
git clone https://github.com/your-repo/partimorph.git
cd partimorph
pip install -e .
```

## References

- **Wadell, H. (1932)**: Volume, shape, and roundness of rock particles. *The Journal of Geology*.
- **Riley, N. A. (1941)**: Projection sphericity. *Journal of Sedimentary Research*.
- **ISO 9276-6**: Representation of results of particle size analysis — Part 6: Descriptive and quantitative representation of particle shape and morphology.
