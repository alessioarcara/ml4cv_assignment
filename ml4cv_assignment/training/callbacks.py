from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.decomposition import PCA
from torch import Tensor
from torchinfo import summary

import wandb
from ml4cv_assignment.data.cityscapes import CityscapesDataset
from ml4cv_assignment.data.streethazards import StreetHazards
from ml4cv_assignment.utils.palettes import CITYSCAPES_COLORS, STREETHAZARD_COLORS
from ml4cv_assignment.utils.typings import DatasetType, Stage
from ml4cv_assignment.utils.visualize import apply_colormap, color

if TYPE_CHECKING:
    from ml4cv_assignment.training.trainer import Trainer


class Callback(ABC):
    def on_train_start(self, trainer: "Trainer"): ...
    def on_train_end(self, trainer: "Trainer"): ...
    def on_eval_end(self, trainer: "Trainer"): ...


class ModelMonitorCallback(Callback):
    def __init__(self, history_key: str, minimize: bool):
        super().__init__()
        self.history_key = history_key
        self.best = float("inf") if minimize else 0.0
        self.minimize = minimize

    def on_eval_end(self, trainer: "Trainer") -> bool:
        val = trainer.history.get(self.history_key)
        if val is None:
            logger.warning(f"{self.history_key} not found in history; skipping.")
            return False

        try:
            val = float(val)
        except ValueError:
            logger.warning(f"{self.history_key} value {val} is not a float; skipping.")
            return False

        improved = val < self.best if self.minimize else val > self.best
        if improved:
            self.best = val
            return True

        return False


class EarlyStoppingCallback(ModelMonitorCallback):
    def __init__(self, history_key: str, minimize: bool, patience: int):
        super().__init__(history_key, minimize)
        self.patience = patience
        self.counter = 0

    def on_eval_end(self, trainer: "Trainer") -> bool:
        improved = super().on_eval_end(trainer)

        if improved:
            self.counter = 0
            logger.info("Improved. Counter reset.")
        else:
            self.counter += 1
            logger.info(f"No improvement for {self.counter}/{self.patience}.")

            if self.counter >= self.patience:
                trainer.stop_training = True

        return improved


class ModelSavingCallback(ModelMonitorCallback):
    def __init__(self, history_key: str, minimize: bool, out_dir: str):
        super().__init__(history_key, minimize)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_model: Optional[Dict[str, Tensor]] = None

    def on_eval_end(self, trainer: "Trainer") -> bool:
        improved = super().on_eval_end(trainer)

        if improved:
            self.best_model = trainer.model.state_dict()

        return improved

    def on_train_end(self, trainer: "Trainer") -> None:
        if wandb.run is None:
            logger.warning("No active wandb run detected; skipping model upload.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{trainer.config.wandb_base_run_name}_{timestamp}_{self.history_key}_{self.best:0.4f}"

        base_name = base_name.replace(" ", "_").replace("/", "-")  # sanitize filename

        checkpoint_path = self.out_dir / f"{base_name}.pth"

        torch.save(self.best_model, checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

        artifact = wandb.Artifact(name=base_name, type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.run.log_artifact(artifact)
        logger.info(f"Uploaded model artifact to wandb: {base_name}")


class ModelSummaryCallback(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer: "Trainer") -> None:
        model = trainer.model
        train_loader = trainer.get_loader(Stage.TRAIN)
        assert train_loader is not None

        # dummy_batch = next(iter(train_loader))
        summary(model, verbose=1)


class VisualizeSegmentationResultsCallback(Callback):
    """
    Log a side-by-side comparison of true vs predicted segmentation masks.
    """

    def __init__(
        self, num_samples: int, dataset_type: DatasetType = DatasetType.STREETHAZARDS
    ) -> None:
        self.num_samples = num_samples
        self.eval_step = 0

        if dataset_type == DatasetType.STREETHAZARDS:
            self.palette = STREETHAZARD_COLORS
        elif dataset_type == DatasetType.CITYSCAPES:
            self.palette = CITYSCAPES_COLORS

    def on_train_start(self, trainer: "Trainer") -> None:
        self.table = wandb.Table(columns=["step", "image"], log_mode="INCREMENTAL")

    def on_eval_end(self, trainer: "Trainer") -> None:
        val_loader = trainer.get_loader(Stage.VAL)
        if val_loader is None:
            logger.warning("Validation loader not available; skipping visualization.")
            return

        batch = next(iter(val_loader))
        inputs: Dict[str, Tensor] = trainer._prepare_input(batch)  # type: ignore

        with torch.inference_mode():
            outputs = trainer.model(inputs)

        imgs = inputs["pixel_values"][: self.num_samples]
        gt_masks = inputs["orig_masks"][: self.num_samples]
        pred_masks = outputs["preds"][: self.num_samples]
        ood_scores = outputs.get("ood_score")

        if ood_scores is not None:
            ood_scores = ood_scores[: self.num_samples]
        else:
            ood_scores = [None] * len(imgs)

        for img, gt_mask, pred_mask, ood_score in zip(
            imgs,
            gt_masks,
            pred_masks,
            ood_scores,
        ):
            img_np = trainer.denormalize(img)
            true_colored = color(gt_mask.cpu().numpy(), self.palette)
            pred_colored = color(pred_mask.cpu().numpy(), self.palette)

            images_to_concat = [img_np, true_colored, pred_colored]

            if ood_score is not None:
                heatmap = apply_colormap(ood_score.cpu().numpy(), cmap_name="jet")
                images_to_concat.append(heatmap)

            comparison = np.concatenate(images_to_concat, axis=1)
            self.table.add_data(self.eval_step, wandb.Image(comparison))

        wandb.log({"Segmentation Results": self.table})

        self.eval_step += 1


class PixelEmbeddingsCallback(Callback):
    """
    Log PCA visualization of pixels embeddings for the first image in a batch.
    """

    def __init__(
        self,
        anchors_magnitude: float,
        min_num_pixels: int,
        dataset_type: DatasetType = DatasetType.STREETHAZARDS,
    ) -> None:
        self.anchors_magnitude = anchors_magnitude
        self.min_num_pixels = min_num_pixels

        if dataset_type == DatasetType.STREETHAZARDS:
            self.id_to_label_map = StreetHazards.id_to_label_map()
        else:
            self.id_to_label_map = CityscapesDataset.id_to_label_map()

    def on_eval_end(self, trainer: "Trainer") -> None:
        val_loader = trainer.get_loader(Stage.VAL)
        if val_loader is None:
            logger.warning("Validation loader not available; skipping visualization.")
            return

        batch = next(iter(val_loader))
        inputs: Dict[str, Tensor] = trainer._prepare_input(batch)  # type: ignore

        with torch.inference_mode():
            outputs = trainer.model(inputs, return_preds=False)

        logits = outputs["logits"]
        true = inputs["orig_masks"]

        # Take first image in batch
        logits0 = logits[0].detach().cpu()  # [C, H, W]
        labels0 = true[0].detach().cpu().numpy().flatten()  # [H*W]

        C = logits0.shape[0]
        embeddings = logits0.permute(1, 2, 0).reshape(-1, C).cpu().numpy()  # [H*W, C]

        # Sample pixels from each class
        indices: List[int] = []
        for cls in np.unique(labels0):
            cls_indices = np.where(labels0 == cls)[0]
            if cls_indices.size == 0:
                continue
            n_samples = min(len(cls_indices), self.min_num_pixels)
            chosen = np.random.choice(cls_indices, size=n_samples, replace=False)
            indices.extend(chosen.tolist())

        selected_embeddings = embeddings[indices]
        selected_classes = labels0[indices].astype(int)

        # Anchors: one anchor vector per class scaled by magnitude
        anchors = np.eye(C) * self.anchors_magnitude
        combined_embeddings = np.vstack([selected_embeddings, anchors])  # [N + C, C]

        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(combined_embeddings)  # [N + C, 2]

        data_points = points_2d[:-C]
        anchor_points = points_2d[-C:]

        # Plot pixel embeddings
        plt.figure(figsize=(12, 8))
        for cls in np.unique(selected_classes):
            mask = selected_classes == cls
            class_name = self.id_to_label_map.get(cls, f"Class {cls}")
            plt.scatter(
                data_points[mask, 0],
                data_points[mask, 1],
                label=class_name,
                alpha=0.6,
                s=10,
            )

        # Plot anchors
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
            class_name = self.id_to_label_map.get(cls, f"Class {cls}")
            plt.annotate(
                class_name,
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
