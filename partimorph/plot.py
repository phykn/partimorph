import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon


def plot_analysis_results(
    mask: np.ndarray,
    results: dict,
    figsize: tuple[int, int] = (8, 8),
    title_prefix: str = "Analysis Results",
) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mask, cmap="gray", alpha=0.3)

    if results.get("sphericity"):
        in_circle = results["sphericity"]["inscribed"]
        circ_in = Circle(
            xy = (in_circle["x"], in_circle["y"]),
            radius = in_circle["r"],
            color = "blue",
            fill = False,
            label = "Inscribed Circle",
        )
        ax.add_patch(circ_in)
        ax.scatter(in_circle["x"], in_circle["y"], color="blue", marker="+")

        en_circle = results["sphericity"]["enclosing"]
        circ_en = Circle(
            xy = (en_circle["x"], en_circle["y"]),
            radius = en_circle["r"],
            color = "red",
            fill = False,
            label = "Enclosing Circle",
        )
        ax.add_patch(circ_en)
        ax.scatter(en_circle["x"], en_circle["y"], color="red", marker="x")

    if results.get("ellipse"):
        bbox = results["ellipse"]["bbox"]
        poly = Polygon(
            bbox,
            closed = True,
            color = "green",
            fill = False,
            label = "Ellipse BBox",
        )
        ax.add_patch(poly)

    title_str = f"{title_prefix}\n"
    metrics = []

    if "roundness" in results:
        metrics.append(f"Roundness: {results['roundness']['val']:.2f}")
    if "circularity" in results:
        metrics.append(f"Circularity: {results['circularity']['val']:.2f}")
    if "sphericity" in results:
        metrics.append(f"Sphericity: {results['sphericity']['val']:.2f}")
    if "ellipse" in results and results["ellipse"] is not None:
        metrics.append(f"Aspect Ratio: {results['ellipse']['val']:.2f}")

    ax.set_title(title_str + ", ".join(metrics))
    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
