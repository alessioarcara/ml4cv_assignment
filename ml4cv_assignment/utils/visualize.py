"""
Source: https://github.com/hendrycks/anomaly-seg/issues/15#issuecomment-890300278
"""

import math
import random
from typing import Any, Callable, List, Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from loguru import logger
from PIL import Image

from ml4cv_assignment.models.base_model import BaseModel as Model
from ml4cv_assignment.utils.misc import resolve_device
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


def show_injected_outlier(sample, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualizes an image with an injected outlier and its corresponding segmentation mask.

    Args:
        sample: A tuple containing the image and its ground truth mask.
        figsize: Size of the figure for visualization.
    """
    img, gt_mask = sample
    label_vis = color(gt_mask, COLORS)

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].imshow(img)
    axs[0].set_title("Image (Outlier Injected)")
    axs[0].axis("off")

    axs[1].imshow(label_vis)
    axs[1].set_title("Segmentation Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_augmentations(
    dataset: Any,
    denorm: Callable[[torch.Tensor], np.ndarray],
    idx: Optional[int] = None,
    num_samples: int = 9,
    cols: int = 3,
) -> None:
    """
    Visualizes original image and its augmented versions from the dataset.

    Args:
        dataset: Torch Dataset object.
        denorm: Function to convert tensor to numpy image (H, W, C).
        idx: Index of the image. If None, random.
        num_samples: Total images to show.
        cols: Number of columns.
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    orig_img, _ = dataset.__getitem__(idx, apply_transforms=False)
    imgs_to_show = [orig_img]

    for _ in range(num_samples - 1):
        aug_img, _ = dataset[idx]
        imgs_to_show.append(denorm(aug_img))

    rows = math.ceil(num_samples / cols)
    _, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        if i < len(imgs_to_show):
            ax.imshow(imgs_to_show[i])
            ax.set_title("Original" if i == 0 else f"Augmented {i}")
            ax.axis("off")
        else:
            ax.set_visible(False)  # Hide unused subplots

    plt.suptitle(f"Original and Augmented Images (Idx: {idx})", fontsize=14)
    plt.tight_layout()
    plt.show()


def _plot_lines(data: List[dict]) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    for run in data:
        name = run["run_name"]
        ax1.plot(run["val/loss_epoch"], label=name, linewidth=2)
        ax2.plot(run["val/mIoU_epoch"], label=name, linewidth=2)

    ax1.set_title("Val Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()
    ax2.set_title("Val mIoU Comparison")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()
    plt.tight_layout()
    plt.show()


def _plot_radar(categories: List[str], data: dict) -> None:
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


def compare_runs(
    retriever: WandBRetriever,
    run_ids: List[str],
    classes: List[str],
) -> None:
    """Compares multiple runs using line and radar plots."""
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


def interactive_inference_visualizer(
    model: Model,
    dataset: Any,
    denorm: Callable[[torch.Tensor], np.ndarray],
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Interactive visualizer for model inference on a dataset.

    Args:
        model: The segmentation model.
        dataset: Dataset returning (image, ground_truth_mask) tuples.
        denorm: Function to remove normalization and convert tensor to numpy image (H, W, C).
        device: Device to run inference on.
    """
    device = resolve_device(device)
    model = model.to(device)
    model.eval()

    idx_slider = widgets.IntSlider(
        value=0, min=0, max=len(dataset) - 1, step=1, description="Index:"
    )

    out = widgets.Output()

    @torch.inference_mode()
    def update_plot(change):
        idx = change["new"]

        img, gt_mask = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)  # [1, C, H, W]

        inputs = {
            "pixel_values": input_tensor,
        }

        outputs = model(inputs, return_preds=True)

        closed_set_preds = outputs["preds"]
        ood_score = outputs["ood_score"]

        with out:
            clear_output(wait=True)

            # Layout 1x4: Img | GT Mask | Closed-set Preds | OOD Score
            fig, axes = plt.subplots(1, 4, figsize=(20, 6))

            img_vis = denorm(img)
            # Original Image
            axes[0].imshow(img_vis)
            axes[0].set_title("Image")

            # Ground-truth Mask
            gt_vis = color(gt_mask.cpu().squeeze(), COLORS)
            axes[1].imshow(gt_vis)
            axes[1].set_title("Ground-truth mask")

            # Closed-set Predictions
            pred_vis = closed_set_preds.cpu().squeeze()
            axes[2].imshow(color(pred_vis, COLORS))
            axes[2].set_title("Closed-set mask")

            # OOD Score
            ood_vis = ood_score.cpu().squeeze()
            im_ood = axes[3].imshow(ood_vis, cmap="viridis")
            axes[3].set_title("Anomaly score")

            plt.colorbar(im_ood, ax=axes[3], fraction=0.046, pad=0.04)

            for a in axes:
                a.axis("off")

            plt.tight_layout()
            plt.show()

            plt.close(fig)

    idx_slider.observe(lambda change: update_plot(change), names="value")
    display(widgets.VBox([idx_slider, out]))
    update_plot({"new": idx_slider.value})
