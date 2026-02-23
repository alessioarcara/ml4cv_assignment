"""
Source: https://github.com/hendrycks/anomaly-seg/issues/15#issuecomment-890300278
"""

import math
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from ml4cv_assignment.utils.charts import (
    plot_latent_spaces,
    plot_radar_chart,
    plot_training_curves,
)
from ml4cv_assignment.utils.misc import resolve_device
from ml4cv_assignment.utils.palettes import STREETHAZARD_COLORS as COLORS
from ml4cv_assignment.utils.tables import print_eval_results, print_metrics_table
from ml4cv_assignment.utils.typings import MetricModality
from ml4cv_assignment.utils.wandb_retriever import WandBRetriever

if TYPE_CHECKING:
    from ml4cv_assignment.config import Config


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


def show_runs_comparison(
    retriever: WandBRetriever,
    run_ids: List[str],
    classes: List[str],
    mode: Literal["closed", "open"] = "closed",
    force_download: bool = False,
    show_latent_space: bool = False,
    show_plots: bool = True,
    include_loss: bool = True,
) -> None:
    """Compares multiple runs using line and radar plots."""
    num_classes = len(classes) - 1  # Exclude anomaly class

    metrics_config = {
        "val/mIoU_epoch": MetricModality.FULL,
        **{
            f"val/IoU_class_{i}_epoch": MetricModality.SINGLE
            for i in range(num_classes)
        },
    }

    if include_loss:
        metrics_config["val/loss_epoch"] = MetricModality.FULL

    if mode == "open":
        metrics_config["val/OoDAUPR_epoch"] = MetricModality.FULL

    if show_latent_space:
        metrics_config["pixel_embeddings_pca"] = MetricModality.MEDIA

    data = retriever.get_metrics(
        run_ids=run_ids,
        metrics_config=metrics_config,
        reference_metric="val/mIoU_epoch",
        mode="max",
        force_download=force_download,
    )

    if not data:
        logger.warning("No valid runs found for comparison.")
        return

    radar_data = {
        run["run_name"]: [run[f"val/IoU_class_{i}_epoch"] for i in range(num_classes)]
        for run in data
    }

    print_metrics_table(data, mode=mode)
    if show_plots:
        plot_training_curves(data)
        plot_radar_chart(classes[:-1], radar_data)
    if show_latent_space:
        plot_latent_spaces(data)


def show_split_results(
    results: Dict[str, float],
    classes: List[str],
    metric_names: List[str] = ["mIoU", "OoDAUPR"],
) -> None:
    """
    Shows a table for key metrics and a radar chart for class-wise IoU.

    Args:
        results: Dictionary containing evaluation results.
        classes: List of class names.
        metric_names: List of substrings to search for in keys.
    """

    def find_val(search_str):
        return next((v for k, v in results.items() if search_str in k), 0.0)

    # --- Metric Table ---
    metric_data = {name: find_val(name) for name in metric_names}
    print_eval_results(metric_data, "Quantitative Results")

    # --- Radar Chart ---
    plot_classes = classes[:-1]
    radar_values = [find_val(f"IoU_class_{i}") for i in range(len(plot_classes))]
    plot_radar_chart(classes[:-1], {"run": radar_values})


def interactive_inference_visualizer(
    cfg: "Config",
    split: Literal["val", "test"],
    device: Optional[Union[str, torch.device]] = None,
    static_preview: bool = True,
    static_preview_index: int = 1025,
):
    """
    Interactive visualizer for model inference on a dataset.

    Args:
        cfg: Configuration object containing model and dataset.
        split: Dataset split to visualize ('val' or 'test').
        device: Device to run inference on.
    """
    device = resolve_device(device)

    model = cfg.model.to(device)
    model.eval()

    dataset = cfg.val_dataset if split == "val" else cfg.test_dataset
    denorm = cfg.training.get_denormalize()

    def visualize_single_sample(idx: int) -> plt.Figure:
        img, gt_mask = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)  # [1, C, H, W]

        inputs = {"pixel_values": input_tensor}

        with torch.inference_mode():
            outputs = model(inputs, return_preds=True)

        closed_set_preds = outputs["preds"]
        ood_score = outputs["ood_score"]

        # Layout 1x4: Img | GT Mask | Closed-set Preds | OOD Score
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        # 1. Original Image
        img_vis = denorm(img)
        axes[0].imshow(img_vis)
        axes[0].set_title("Image")

        # 2. Ground-truth Mask
        gt_vis = color(gt_mask.cpu().squeeze(), COLORS)
        axes[1].imshow(gt_vis)
        axes[1].set_title("Ground-truth mask")

        # 3. Closed-set Predictions
        pred_vis = closed_set_preds.cpu().squeeze()
        axes[2].imshow(color(pred_vis, COLORS))
        axes[2].set_title("Closed-set mask")

        # 4. OOD Score
        ood_vis = ood_score.cpu().squeeze()
        im_ood = axes[3].imshow(ood_vis, cmap="viridis")
        axes[3].set_title("Anomaly score")

        # Colorbar
        divider = make_axes_locatable(axes[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_ood, cax=cax)

        for a in axes:
            a.axis("off")

        plt.tight_layout()
        return fig

    if static_preview:
        print(f"Static Preview (Index {static_preview_index}) for split '{split}':")
        fig = visualize_single_sample(static_preview_index)
        plt.show(fig)

    print("\nInteractive Explorer:")

    idx_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(dataset) - 1,
        step=1,
        description="Index:",
        continuous_update=False,
    )

    out = widgets.Output()

    @torch.inference_mode()
    def update_plot(change):
        idx = change["new"]
        with out:
            clear_output(wait=True)
            fig = visualize_single_sample(idx)
            plt.show(fig)
            plt.close(fig)

    idx_slider.observe(lambda change: update_plot(change), names="value")
    display(widgets.VBox([idx_slider, out]))
    update_plot({"new": idx_slider.value})
