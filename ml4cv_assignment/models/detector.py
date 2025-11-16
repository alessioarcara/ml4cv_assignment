from typing import Any, Dict

import torch
import torch.nn as nn
from loguru import logger
from tqdm.notebook import tqdm

from ml4cv_assignment.models.deeplab_fpn import DeepLabFPN


class RunningCenters(nn.Module):
    def __init__(self, n_classes: int):
        super(RunningCenters, self).__init__()
        self.num_classes = n_classes
        self.register_buffer(
            "running_centers", torch.zeros(self.num_classes, self.num_classes)
        )
        self.register_buffer("running_counts", torch.zeros(self.num_classes))

    @property
    def centers(self) -> torch.Tensor:
        return self.running_centers / self.running_counts.clamp(min=1).unsqueeze(1)

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        for cls in range(self.num_classes):
            class_mask = labels == cls
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                class_sum = class_embeddings.sum(dim=0)
                self.running_centers[cls] += class_sum
                self.running_counts[cls] += class_mask.sum().float()

    def fit(self, model, loader, device):
        self.to(device)
        model = model.to(device).eval()

        self.reset()

        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc="Computing class centers"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).view(-1)

                logits = model(imgs)  # [B, C, H, W]
                embeds = logits.permute(0, 2, 3, 1).reshape(
                    -1, self.num_classes
                )  # [B*H*W, C]

                self.update(embeds, masks)

        return self

    def reset(self):
        self.running_centers.zero_()
        self.running_counts.zero_()


class OpenSetSegmenter(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        model: DeepLabFPN,
        n_classes: int,
        alpha: float = 0.5,
        T: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(OpenSetSegmenter, self).__init__()
        model.eval()
        self.model = model
        self.n_classes = n_classes
        self.xi = config["losses"]["objectosphere"]["xi"]
        self.alpha = alpha
        self.T = T
        self.device = device

        self.register_buffer(
            "anchors",
            torch.eye(n_classes, device=device)
            * config["losses"]["prototypical_triplet"]["anchors_magnitude"],
        )

    @torch.no_grad()
    def fit(self, loader: torch.utils.data.DataLoader) -> None:
        rc = RunningCenters(self.n_classes).fit(self.model, loader, self.device)
        self.anchors.copy_(rc.centers)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        closed_set_preds = logits.argmax(dim=1)
        open_set_probs = self.computeAnomalyProbabilities(logits)
        return closed_set_preds, open_set_probs

    def computeAnomalyProbabilities(self, logits: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        embeds = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [(B·H·W), C]

        # Feature-norm score
        norms_sq = embeds.norm(p=2, dim=1).pow(2)  # [(B·H·W)]
        feat_score = (1.0 - norms_sq / self.xi).clamp(min=0.0)  # [(B·H·W)]

        # Prototype distance score
        dists = torch.cdist(embeds, self.anchors, p=2)  # [(B·H·W), K]
        softmin = torch.softmax(-dists / self.T, dim=1)
        max_prob, _ = softmin.max(dim=1)
        dist_score = 1.0 - max_prob  # [(B·H·W)]

        anomaly_score = self.alpha * feat_score + (1 - self.alpha) * dist_score
        return anomaly_score.reshape(B, H, W)

    def save_anchors(self, path: str) -> None:
        torch.save(self.anchors, path)
        logger.info("Anchors saved to {}", path)

    def load_anchors(self, path: str) -> None:
        self.anchors = torch.load(path, map_location=self.device)
        logger.info("Anchors loaded from {}", path)
