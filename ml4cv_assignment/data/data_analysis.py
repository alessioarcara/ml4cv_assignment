from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from tqdm import tqdm

from ml4cv_assignment.data.streethazards import StreetHazards
from ml4cv_assignment.utils.typings import ClassIdCounter
from ml4cv_assignment.utils.visualize import COLORS, color


@dataclass
class DatasetStats:
    image_sizes: Set[Tuple[int, int]]
    pixel_counts: ClassIdCounter
    image_counts: ClassIdCounter
    rgb_means: np.ndarray
    rgb_stds: np.ndarray


def analyze_dataset(dataset: StreetHazards) -> "DatasetStats":
    """
    Computes:
      - Image sizes (H, W)
      - Pixel counts per class
      - Image counts per class
      - Per-image mean values for each RGB channel
      - Per-image std values for each RGB channel
    """
    sizes = set()
    pixel_counts: ClassIdCounter = Counter()
    image_counts: ClassIdCounter = Counter()
    rgb_means: list[np.ndarray] = []
    rgb_stds: list[np.ndarray] = []

    for img, mask in tqdm(dataset, desc="Analyzing dataset", unit="img"):
        sizes.add(img.shape[:2])

        vals, cnts = np.unique(mask, return_counts=True)

        pixel_counts.update(dict(zip(vals, cnts)))  # per-pixel totals
        image_counts.update(vals.tolist())  # per-image presence

        img_f = img.astype(np.float32)
        img_f /= 255.0
        rgb_means.append(img_f.mean(axis=(0, 1)))
        rgb_stds.append(img_f.std(axis=(0, 1)))

    rgb_means_np = np.stack(rgb_means)
    rgb_stds_np = np.stack(rgb_stds)

    return DatasetStats(
        image_sizes=sizes,
        pixel_counts=pixel_counts,
        image_counts=image_counts,
        rgb_means=rgb_means_np,
        rgb_stds=rgb_stds_np,
    )


def compute_class_heatmap(
    dataset: Iterable[Tuple[Any, np.ndarray]],
    height: int = 720,
    width: int = 1280,
    num_classes: int = 13,
) -> np.ndarray:
    """
    Compute a heatmap where each channel contains the number of times that class
    appeared at each position.
    """
    heatmap = np.zeros((height, width, num_classes), dtype=np.int32)

    for _, mask in tqdm(dataset, desc="Computing heatmap", unit="img"):
        rows, cols = np.indices((height, width))
        heatmap[rows, cols, mask] += 1

    return heatmap


def plot_class_counts(
    train_class_counts: ClassIdCounter,
    val_class_counts: ClassIdCounter,
    id_to_label_map: Optional[Dict[int, str]] = None,
    title: str = "",
) -> None:
    df_train = pl.DataFrame(
        {
            "class_id": list(train_class_counts.keys()),
            "count": list(train_class_counts.values()),
            "split": ["train"] * len(train_class_counts),
        }
    )

    df_val = pl.DataFrame(
        {
            "class_id": list(val_class_counts.keys()),
            "count": list(val_class_counts.values()),
            "split": ["val"] * len(val_class_counts),
        }
    )

    df = pl.concat([df_train, df_val], how="vertical")

    df = df.with_columns(
        (pl.col("count") / pl.sum("count").over("split")).alias("count")
    )

    if id_to_label_map is not None:
        df = df.with_columns(
            pl.col("class_id")
            .cast(pl.Utf8)
            .replace(id_to_label_map)
            .alias("class_label")
        )
        x_col = "class_label"
    else:
        x_col = "class_id"

    pdf = pd.DataFrame(df.to_dicts())
    ax = sns.barplot(data=pdf, x=x_col, y="count", hue="split", dodge=True)
    ax.set_title(f"{title} count per class: train vs val")
    ax.set_xlabel(x_col)
    ax.set_ylabel("perc")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.show()


def plot_rgb_means(rgb_means: np.ndarray):
    for i, c in enumerate(["r", "g", "b"]):
        sns.histplot(rgb_means[:, i], color=c, label=c.upper(), alpha=0.5, kde=True)
    plt.xlabel("Channel mean value")
    plt.ylabel("Count")
    plt.title("Distribution of mean values per RGB channel")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_class_heatmaps(
    train_heatmap: np.ndarray, val_heatmap: np.ndarray, id_to_label_map=None
):
    """
    Plots train and validation position heatmaps side by side for each class.
    """
    num_classes = train_heatmap.shape[2]

    class_labels = [id_to_label_map.get(i, f"Class {i}") for i in range(num_classes)]

    _, axes = plt.subplots(nrows=num_classes, ncols=2, figsize=(10, 4 * num_classes))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i in range(num_classes):
        # Normalize for visualization
        train_img = train_heatmap[:, :, i]
        val_img = val_heatmap[:, :, i]

        train_img = train_img / train_img.sum()
        val_img = val_img / val_img.sum()

        ax_train = axes[i, 0]
        ax_val = axes[i, 1]

        ax_train.imshow(train_img, cmap="gray")
        ax_train.set_title(f"Train - {class_labels[i]}")
        ax_train.axis("off")

        ax_val.imshow(val_img, cmap="gray")
        ax_val.set_title(f"Validation - {class_labels[i]}")
        ax_val.axis("off")

    plt.tight_layout()
    plt.show()


def browse_dataset(
    dataset: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Creates an interactive slider to browse images and ground truth masks from a
    dataset.
    """

    def show_item(idx: int):
        img_np, mask_gt = dataset[idx]

        mask_plot = color(mask_gt, COLORS)

        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.imshow(img_np)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(mask_plot)
        plt.title("Ground truth Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    idx_slider = widgets.IntSlider(
        value=0, min=0, max=len(dataset) - 1, step=1, description="Index:"
    )

    widgets.interact(show_item, idx=idx_slider)
