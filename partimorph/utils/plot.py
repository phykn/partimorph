import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Mapping
from matplotlib.patches import Circle, Ellipse, Polygon


def plot_analysis_results(
    mask: np.ndarray,
    results: Mapping[str, Any],
    figsize: tuple[int, int] = (8, 8),
    title: str = "Analysis Results",
) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mask, cmap="gray", alpha=0.3)

    sphericity_data = results.get("sphericity")
    if sphericity_data:
        inscribed = sphericity_data["inscribed"]
        ax.add_patch(
            Circle(
                (inscribed["x"], inscribed["y"]),
                inscribed["r"],
                color="blue",
                fill=False,
                label="Inscribed",
            )
        )
        ax.scatter(inscribed["x"], inscribed["y"], color="blue", marker="+")

        enclosing = sphericity_data["enclosing"]
        ax.add_patch(
            Circle(
                (enclosing["x"], enclosing["y"]),
                enclosing["r"],
                color="red",
                fill=False,
                label="Enclosing",
            )
        )
        ax.scatter(enclosing["x"], enclosing["y"], color="red", marker="x")

    aspect_ratio_data = results.get("aspect_ratio")
    if aspect_ratio_data:
        ellipse = aspect_ratio_data.get("ellipse")
        if ellipse:
            ax.add_patch(
                Ellipse(
                    (ellipse["x"], ellipse["y"]),
                    ellipse["w"],
                    ellipse["h"],
                    angle=-ellipse["angle"],
                    color="yellow",
                    fill=False,
                    label="Fitted Ellipse",
                )
            )
            if ellipse.get("bbox"):
                ax.add_patch(
                    Polygon(
                        ellipse["bbox"],
                        closed=True,
                        color="green",
                        fill=False,
                        label="Bounding Box",
                    )
                )

    metric_labels = {
        "roundness": "Roundness",
        "circularity": "Circularity",
        "sphericity": "Sphericity",
        "aspect_ratio": "Aspect Ratio",
    }

    stats = []
    for key, label in metric_labels.items():
        data = results.get(key)
        if data and "val" in data:
            stats.append(f"{label}: {data['val']:.4f}")

    ax.set_title(f"{title}\n{', '.join(stats)}")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
