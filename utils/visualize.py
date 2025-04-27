"""
Source: https://github.com/hendrycks/anomaly-seg/issues/15#issuecomment-890300278
"""

import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE

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
) -> Image.Image:
    if isinstance(annot_or_mask, str):
        mask = np.array(Image.open(annot_or_mask))
    else:
        mask = annot_or_mask

    img_new = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for index, color in enumerate(colors):
        img_new[mask == index] = color

    return Image.fromarray(img_new, "RGB")


def visualize_augmentations(dataset, data_transforms, denorm, num_samples=6, cols=3):
    img, _ = dataset.__getitem__(0, apply_transforms=False)  # original image

    imgs = [img]
    for _ in range(num_samples - 1):
        augmented = data_transforms["train"](image=img)["image"]
        imgs.append(denorm(augmented))

    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    axes = axes.flatten() if rows > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < num_samples:
            ax.imshow(imgs[i])
            ax.axis("off")
        else:
            ax.set_visible(False)  # Hide unused subplots

    plt.suptitle("Original and Augmented Images", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_added_outlier(sample, denorm, figsize=(12, 8)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    img = denorm(sample[0])
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis("off")

    label_vis = color(sample[1], COLORS)
    axs[1].imshow(label_vis)
    axs[1].set_title("Labels")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_centers(centers):
    if isinstance(centers, torch.Tensor):
        centers = centers.detach().cpu().numpy()
    if not isinstance(centers, np.ndarray):
        raise TypeError(
            f"`centers` must be torch.Tensor or np.ndarray, got {type(centers)!r}"
        )
    tsne = TSNE(n_components=2, random_state=42, perplexity=centers.shape[0] - 1)
    centers_2d = tsne.fit_transform(centers)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        s=200,
        c=range(centers.shape[0]),
        cmap="tab20",
    )

    for i in range(centers.shape[0]):
        plt.annotate(
            f"{i}",
            (centers_2d[i, 0], centers_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    plt.title(f"{centers.shape[0]} Class Centers (t-SNE visualization)")
    plt.axis("off")
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    annot_path = "/path/to/input/annotation"
    segm_map = color(annot_path, COLORS)
    segm_map.save("/path/to/output/map")
