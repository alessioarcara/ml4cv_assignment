"""
Source: https://github.com/hendrycks/anomaly-seg/issues/15#issuecomment-890300278
"""

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image

from ml4cv_assignment.utils.typings import MetricModality
from ml4cv_assignment.utils.wandb_retriever import WandBRetriever

COLORS = np.array(
    [
        [0, 0, 0],  # unlabeled    =   0,
        [70, 70, 70],  # building     =   1,
        [190, 153, 153],  # fence        =   2,
        [250, 170, 160],  # other        =   3,
        [220, 20, 60],  # pedestrian   =   4,
        [153, 153, 153],  # pole         =   5,
        [157, 234, 50],  # road line    =   6,
        [128, 64, 128],  # road         =   7,
        [244, 35, 232],  # sidewalk     =   8,
        [107, 142, 35],  # vegetation   =   9,
        [0, 0, 142],  # car          =  10,
        [102, 102, 156],  # wall         =  11,
        [220, 220, 0],  # traffic sign =  12,
        [60, 250, 240],  # anomaly      =  13,
    ]
)


def color(
    annot_or_mask: Union[str, np.ndarray],
    colors: np.ndarray,
) -> np.ndarray:
    # Load mask if path is provided
    if isinstance(annot_or_mask, str):
        mask = np.array(Image.open(annot_or_mask))
    else:
        mask = annot_or_mask

    img_new = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for index, color in enumerate(colors):
        img_new[mask == index] = color

    return img_new


def apply_colormap(mask: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    cmap = plt.get_cmap(cmap_name)
    colored_rgba = cmap(mask_norm)
    colored_rgb = colored_rgba[..., :3]  # Discard alpha channel
    colored_rgb_uint8 = (colored_rgb * 255).astype(np.uint8)
    return colored_rgb_uint8


def _plot_lines(data: List[dict]):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for run in data:
        name = run["run_name"]
        ax1.plot(run["val/loss_epoch"], label=name, linewidth=2)
        ax2.plot(run["val/mIoU_epoch"], label=name, linewidth=2)

    ax1.set_title("Val Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.set_title("Val mIoU Comparison")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def _plot_radar(categories: List[str], data: dict):
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
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


def compare_runs(
    retriever: WandBRetriever,
    run_ids: List[str],
    classes: List[str],
) -> None:
    num_classes = len(classes) - 1  # Exclude anomaly class

    metrics_config = {
        "val/mIoU_epoch": MetricModality.FULL,
        "val/loss_epoch": MetricModality.FULL,
        **{
            f"val/IoU_class_{i}_epoch": MetricModality.SINGLE
            for i in range(num_classes)
        },
    }

    data = retriever.get_metrics(
        run_ids=run_ids,
        metrics_config=metrics_config,
        reference_metric="val/mIoU_epoch",
        mode="max",
    )

    if not data:
        logger.warning("No valid runs found for comparison.")
        return

    radar_data = {
        run["run_name"]: [run[f"val/IoU_class_{i}_epoch"] for i in range(num_classes)]
        for run in data
    }

    _plot_lines(data)
    _plot_radar(classes[:-1], radar_data)


# def visualize_augmentations(dataset, data_transforms, denorm, num_samples=6, cols=3):
#    img, _ = dataset.__getitem__(0, apply_transforms=False)  # original image
#
#    imgs = [img]
#    for _ in range(num_samples - 1):
#        augmented = data_transforms["train"](image=img)["image"]
#        imgs.append(denorm(augmented))
#
#    rows = math.ceil(num_samples / cols)
#    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
#
#    axes = axes.flatten() if rows > 1 else [axes]
#
#    for i, ax in enumerate(axes):
#        if i < num_samples:
#            ax.imshow(imgs[i])
#            ax.axis("off")
#        else:
#            ax.set_visible(False)  # Hide unused subplots
#
#    plt.suptitle("Original and Augmented Images", fontsize=14)
#    plt.tight_layout()
#    plt.show()
#
#
# def visualize_added_outlier(sample, denorm, figsize=(12, 8)):
#    fig, axs = plt.subplots(1, 2, figsize=figsize)
#
#    img = denorm(sample[0])
#    axs[0].imshow(img)
#    axs[0].set_title("Image")
#    axs[0].axis("off")
#
#    label_vis = color(sample[1], COLORS)
#    axs[1].imshow(label_vis)
#    axs[1].set_title("Labels")
#    axs[1].axis("off")
#
#    plt.tight_layout()
#    plt.show()
#
#


#
# def visualize_predictions(segmenter, denorm, dataset, device, threshold: float = 0.5):
#    def show_prediction(idx, threshold_value):
#        img, mask = dataset[idx]
#        img = img.unsqueeze(0).to(device)
#
#        closed_set_preds, open_set_probs = segmenter(img)
#        open_set_mask = open_set_probs > threshold_value
#
#        open_set_preds = closed_set_preds.clone()
#        open_set_preds[open_set_mask.squeeze(1)] = 13
#
#        plt.figure(figsize=(18, 10))
#
#        plt.subplot(231)
#        plt.imshow(denorm(img))
#        plt.title("Image")
#        plt.axis("off")
#
#        plt.subplot(232)
#        plt.imshow(color(mask.cpu().squeeze(), COLORS))
#        plt.title("Ground-truth mask")
#        plt.axis("off")
#
#        plt.subplot(233)
#        plt.imshow(color(closed_set_preds.cpu().squeeze(), COLORS))
#        plt.title("Closed-set mask")
#        plt.axis("off")
#
#        plt.subplot(234)
#        plt.imshow(open_set_probs.cpu().squeeze(), cmap="viridis")
#        plt.title("Anomaly score")
#        plt.axis("off")
#
#        plt.subplot(235)
#        plt.imshow(color(open_set_preds.cpu().squeeze(), COLORS))
#        plt.title("Open-set mask")
#        plt.axis("off")
#
#        plt.tight_layout()
#        plt.show()
#
#    idx_slider = widgets.IntSlider(
#        value=0, min=0, max=len(dataset) - 1, step=1, description="Indice:"
#    )
#
#    threshold_slider = widgets.FloatSlider(
#        value=threshold, min=0.0, max=1.0, step=0.01, description="Threshold:"
#    )
#
#    widgets.interact(show_prediction, idx=idx_slider, threshold_value=threshold_slider)
#
