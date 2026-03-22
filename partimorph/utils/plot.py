import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon


def plot_analysis_results(
    mask: np.ndarray,
    results: dict,
    *,
    figsize: tuple[int, int] = (8, 8),
    title: str = "Analysis Results",
) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mask, cmap="gray", alpha=0.3)

    s_data = results.get("sphericity")
    if s_data:
        inc = s_data["inscribed"]
        ax.add_patch(
            Circle(
                (inc["x"], inc["y"]),
                inc["r"],
                color="blue",
                fill=False,
                label="Inscribed",
            )
        )

        ax.scatter(inc["x"], inc["y"], color="blue", marker="+")
        enc = s_data["enclosing"]
        ax.add_patch(
            Circle(
                (enc["x"], enc["y"]),
                enc["r"],
                color="red",
                fill=False,
                label="Enclosing",
            )
        )

        ax.scatter(enc["x"], enc["y"], color="red", marker="x")

    ar_data = results.get("aspect_ratio")
    if ar_data:
        ellipse = ar_data.get("ellipse")

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

    labels = {
        "roundness": "Roundness",
        "circularity": "Circularity",
        "sphericity": "Sphericity",
        "aspect_ratio": "Aspect Ratio",
    }

    stats = []
    for key, label in labels.items():
        data = results.get(key)
        if data and "val" in data:
            stats.append(f"{label}: {data['val']:.2f}")

    ax.set_title(f"{title}\n{', '.join(stats)}")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
