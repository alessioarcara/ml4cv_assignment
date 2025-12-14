from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ml4cv_assignment.utils.constants import KEY_LOSS, KEY_MIOU, KEY_OOD


def plot_training_curves(data: List[dict]) -> None:
    metrics = [
        (KEY_LOSS, "Val Loss Comparison"),
        (KEY_MIOU, "Val mIoU Comparison"),
    ]

    if KEY_OOD in data[0]:
        metrics.append((KEY_OOD, "Val OoD AUPR Comparison"))

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    for ax, (key, title) in zip(axes, metrics):
        for run in data:
            if key in run:
                ax.plot(run[key], label=run["run_name"], linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_radar_chart(categories: List[str], data: dict) -> None:
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    _, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor("#fafafa")
    ax.grid(color="#b0b0b0", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.spines["polar"].set_visible(False)

    for name, values in data.items():
        values = values + values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)

    plt.title("Class-wise Val IoU Comparison", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()
