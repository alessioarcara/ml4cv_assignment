import random
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torchinfo import summary

import wandb
from ml4cv_assignment.utils.typings import Stage
from ml4cv_assignment.utils.visualize import COLORS, color

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
        base_name = f"{wandb.run.name}_{self.history_key}_{self.best:0.4f}_{timestamp}"
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

        dummy_batch = next(iter(train_loader))
        dummy_input = dummy_batch["pixel_values"]
        input_shape = dummy_input.shape

        summary(model, input_size=input_shape, verbose=1)


class VisualizeSegmentationResultsCallback(Callback):
    """
    Log a side-by-side comparison of true vs predicted segmentation masks.
    """

    def __init__(self, batch_mode: Literal["first", "random"] = "first"):
        assert batch_mode in ["first", "random"], (
            "batch_mode must be 'first' or 'random'"
        )
        self.batch_mode = batch_mode

    def _get_batch(
        self, trainer: "Trainer"
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        val_loader = trainer.get_loader(Stage.VAL)
        if val_loader is None:
            return None

        if self.batch_mode == "first":
            batch = next(iter(val_loader))
        else:  # random
            batch = random.choice(list(val_loader))

        return batch

    def on_eval_end(self, trainer: "Trainer") -> None:
        batch = self._get_batch(trainer)
        if batch is None:
            logger.warning("Validation loader not available; skipping visualization.")
            return

        inputs: Dict[str, Tensor] = trainer._prepare_input(batch)  # type: ignore
        table = wandb.Table(columns=["Segmentation Comparison"])

        with torch.inference_mode():
            outputs = trainer.model(inputs)

        imgs = inputs["pixel_values"]
        gt_masks = inputs["orig_masks"]
        pred_preds = outputs["preds"]

        for img, gt_mask, pred_mask in zip(imgs, gt_masks, pred_preds):
            img_np = trainer.denormalize(img)
            true_colored = color(gt_mask.cpu().numpy(), COLORS)
            pred_colored = color(pred_mask.cpu().numpy(), COLORS)
            comparison = np.concatenate((img_np, true_colored, pred_colored), axis=1)
            table.add_data(wandb.Image(comparison))

        wandb.log({"Segmentation Results": table})


class PixelEmbeddings(Callback):
    """
    Log PCA visualization of pixels embeddings for the first image in a batch.
    """

    def __init__(self):
        pass

    def on_eval_end(self):
        pass


#
#    def log_pixel_embeddings(
#        self,
#        logits: torch.Tensor,
#        pred: torch.Tensor,
#        true: torch.Tensor,
#        min_samples: int = 1000,
#    ) -> None:
#        """
#        Log PCA visualization of pixels embeddings for the first image in a batch.
#        """
#        C = logits.shape[1]
#        embeddings = logits[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
#        labels = true[0].cpu().numpy().flatten()
#        pred = pred[0].cpu().numpy().flatten()
#
#        indices = []
#        for cls in np.unique(labels):
#            cls_indices = np.where(labels == cls)[0]
#            if len(cls_indices) > 0:
#                n_samples = min(len(cls_indices), min_samples)
#                indices.extend(
#                    np.random.choice(cls_indices, size=n_samples, replace=False)
#                )
#
#        anchors = (
#            np.eye(C)
#            * self.config["losses"]["prototypical_triplet"]["anchors_magnitude"]
#        )
#
#        selected_embeddings = embeddings[indices]
#        selected_classes = labels[indices]
#        combined_embeddings = np.vstack([selected_embeddings, anchors])
#
#        pca = PCA(n_components=2, random_state=self.config["seed"])
#        points_2d = pca.fit_transform(combined_embeddings)
#        data_points = points_2d[:-C]
#        anchor_points = points_2d[-C:]
#
#        plt.figure(figsize=(12, 8))
#        for cls in np.unique(selected_classes):
#            mask = selected_classes == cls
#            plt.scatter(
#                data_points[mask, 0],
#                data_points[mask, 1],
#                label=self.class_dict[cls],
#                alpha=0.6,
#                s=10,
#            )
#
#        plt.scatter(
#            anchor_points[:, 0],
#            anchor_points[:, 1],
#            marker="x",
#            s=300,
#            c="red",
#            label="Anchors",
#        )
#
#        # Annotate anchors
#        for cls in range(C):
#            plt.annotate(
#                f"{self.class_dict.get(cls)}",
#                (anchor_points[cls, 0], anchor_points[cls, 1]),
#                xytext=(0, 5),
#                textcoords="offset points",
#                ha="center",
#                fontsize=12,
#                color="black",
#                weight="bold",
#            )
#
#        plt.title("PCA visualization of pixel embeddings with anchors", fontsize=16)
#        plt.axis("off")
#        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
#        plt.tight_layout()
#
#        wandb.log({"pixel_embeddings_pca": wandb.Image(plt)})
#        plt.close()
