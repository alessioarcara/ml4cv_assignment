from contextlib import contextmanager
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

import wandb
from utils.visualize import COLORS, color

from .metrics import Metric


class WandbMonitor:
    def __init__(
        self,
        config: Dict[str, Any],
        metrics: List[Metric],
        denorm: Callable,
        class_dict: Dict[int, str],
    ):
        self.config = config
        self.metrics = metrics
        self.denorm = denorm
        self.class_dict = class_dict

    @contextmanager
    def run_context(self, run_name: str, num_criterions: int):
        wandb.init(
            project=self.config["wandb"]["project"],
            name=run_name,
            config=self.config,
        )

        # Step-level metrics
        wandb.define_metric("train/lr", step_metric="step")
        wandb.define_metric("train/batch_totalloss", step_metric="step")
        for i in range(num_criterions):
            wandb.define_metric(f"train/batch_loss{i}", step_metric="step")

        # Epoch-level metrics
        for split in ["train", "val"]:
            wandb.define_metric(f"{split}/epoch_totalloss", step_metric="epoch")
            for metric in self.metrics:
                metric_name = metric.__class__.__name__
                wandb.define_metric(f"{split}/{metric_name}", step_metric="epoch")

        try:
            yield
        finally:
            wandb.finish()

    def log_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        wandb.log(metrics_dict)

    def log_batch_losses(self, losses: Dict[str, float], step: int) -> None:
        wandb.log({**losses, "step": step})

    def log_training_step(self, loss: float, lr: float, step: int) -> None:
        wandb.log(
            {
                "train/batch_totalloss": loss,
                "train/lr": lr,
                "step": step,
            }
        )

    def log_segmentation_results(
        self,
        imgs: torch.Tensor,
        true: torch.Tensor,
        pred: torch.Tensor,
    ) -> None:
        """
        Log a side-by-side comparison of true vs predicted masks.
        """
        table = wandb.Table(columns=["Comparison"])

        for img, true_mask, pred_mask in zip(imgs, true, pred):
            img_np = self.denorm(img)
            true_np = true_mask.cpu().numpy()
            pred_np = pred_mask.cpu().numpy()

            true_colored = color(true_np, COLORS)
            pred_colored = color(pred_np, COLORS)

            comparison = np.concatenate((img_np, true_colored, pred_colored), axis=1)
            table.add_data(wandb.Image(comparison))

        wandb.log({"Segmentation Results": table})

    def log_pixel_embeddings(
        self,
        logits: torch.Tensor,
        pred: torch.Tensor,
        true: torch.Tensor,
        min_samples: int = 1000,
    ) -> None:
        """
        Log PCA visualization of pixels embeddings for the first image in a batch.
        """
        C = logits.shape[1]
        embeddings = logits[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
        labels = true[0].cpu().numpy().flatten()
        pred = pred[0].cpu().numpy().flatten()

        indices = []
        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            if len(cls_indices) > 0:
                n_samples = min(len(cls_indices), min_samples)
                indices.extend(
                    np.random.choice(cls_indices, size=n_samples, replace=False)
                )

        anchors = (
            np.eye(C)
            * self.config["losses"]["prototypical_triplet"]["anchors_magnitude"]
        )

        selected_embeddings = embeddings[indices]
        selected_classes = labels[indices]
        combined_embeddings = np.vstack([selected_embeddings, anchors])

        pca = PCA(n_components=2, random_state=self.config["seed"])
        points_2d = pca.fit_transform(combined_embeddings)
        data_points = points_2d[:-C]
        anchor_points = points_2d[-C:]

        plt.figure(figsize=(12, 8))
        for cls in np.unique(selected_classes):
            mask = selected_classes == cls
            plt.scatter(
                data_points[mask, 0],
                data_points[mask, 1],
                label=self.class_dict[cls],
                alpha=0.6,
                s=10,
            )

        plt.scatter(
            anchor_points[:, 0],
            anchor_points[:, 1],
            marker="x",
            s=300,
            c="red",
            label="Anchors",
        )

        # Annotate anchors
        for cls in range(C):
            plt.annotate(
                f"{self.class_dict.get(cls)}",
                (anchor_points[cls, 0], anchor_points[cls, 1]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                color="black",
                weight="bold",
            )

        plt.title("PCA visualization of pixel embeddings with anchors", fontsize=16)
        plt.axis("off")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
        plt.tight_layout()

        wandb.log({"pixel_embeddings_pca": wandb.Image(plt)})
        plt.close()
