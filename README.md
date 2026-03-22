# PartiMorph 💎

**PartiMorph** is a high-performance Python library designed for comprehensive and precise particle shape analysis. It provides scientific-grade quantification and visualization of morphology (Roundness, Circularity, Sphericity, Aspect Ratio) from binary masks, built with a focus on **numerical stability** and **computational speed**.

---

## 🚀 Key Features

- **Multi-Metric Analysis**: Extract multiple geometric indicators simultaneously.
    - **Wadell Roundness**: Advanced corners-based curvature analysis using high-speed discretization.
    - **ISO Circularity**: Compactness measurement based on the ISO 9276-6 standard.
    - **Riley Sphericity**: Morphology analysis via inscribed/enclosing circle ratios.
    - **Aspect Ratio**: Precise extraction using tight-fitting ellipse algorithms.
- **Robust Processing Engine (New)**:
    - **LCC Filtering**: Automatic Largest Connected Component extraction to eliminate background noise.
    - **Safe-Padding**: Zero-border protection ensures internal algorithms (Distance Transform, Contour Tracing) never crash on image edges.
- **Advanced Parametric Generation**: 
    - **Fourier Particle Generator**: Generate realistic synthetic particles matching target Sphericity and Roundness.
    - **High-Performance Search**: Adaptive scaling and binary search for target morphology in constant time.
- **Seamless Visualization**: 
    - Overlay inscribed/enclosing circles, OBB (Oriented Bounding Boxes), and fitted ellipses with sub-pixel precision.

---

## 🛠️ Quick Start

```python
import partimorph as pm

# 1. Generate a synthetic particle with target targets
# achieving specific Sphericity and Roundness metrics
mask, info = pm.utils.create_fourier_particle_mask(
    shape=(512, 512), center=(256, 256),
    sphericity=0.75, roundness=0.8,
    base_radius=100, return_info=True
)

# 2. Comprehensive shape analysis (Auto-cleans and Auto-pads)
results = pm.analyze_mask(mask)

# 3. Access achieved results
print(f"Target Roundness: {info['roundness_target']} vs Achieved: {results['roundness']['val']:.4f}")
print(f"Sphericity (Riley): {results['sphericity']['val']:.4f}")

# 4. Professional Visualization
pm.utils.plot_analysis_results(mask, results, title="PartiMorph Analysis")
```

---

## 📊 Metrics Detail

| Metric           | Definition                           | Standard / Reference |
| :--------------- | :----------------------------------- | :------------------- |
| **Roundness**    | Average curvature of surface corners | Wadell (1932)        |
| **Circularity**  | $4\pi \times Area / Perimeter^2$     | ISO 9276-6           |
| **Sphericity**   | $R_{inscribed} / R_{enclosing}$      | Riley (1941)         |
| **Aspect Ratio** | Major Axis / Minor Axis ($\ge 1.0$)  | -                    |

---

## ⚡ Performance & Robustness

PartiMorph is engineered for high-throughput scientific workflows:
- **Numpy Vectorization**: Critical mathematical operations like Shoelace area and max-linear-deviation are fully vectorized, providing up to **10x speedups** over iterative implementations.
- **Edge-Protection**: Internal `crop_mask` logic ensures that boundary pixels never touch image borders, preventing open-contour errors common in other libraries.
- **Clean Analysis**: Built-in Largest Connected Component (LCC) filtering ensures that disjoint noise doesn't skew your statistical results.

---

## 📦 Installation

Requires **Python 3.12+**.

```bash
# Clone and install in editable mode
git clone https://github.com/your-repo/partimorph.git
cd partimorph
pip install -e .
```

---

## 📜 References

- **Wadell, H. (1932)**: Volume, shape, and roundness of rock particles. *The Journal of Geology*.
- **Riley, N. A. (1941)**: Projection sphericity. *Journal of Sedimentary Research*.
- **ISO 9276-6**: Representation of results of particle size analysis — Part 6: Quantitative representation of morphology.
