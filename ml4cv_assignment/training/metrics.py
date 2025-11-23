from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_precision_recall_curve
from torchmetrics.utilities.compute import auc

from ml4cv_assignment.utils.misc import resolve_device
from ml4cv_assignment.utils.typings import MetricResults


class Metric(ABC):
    @abstractmethod
    def update(
        self,
        gt_masks: Tensor,
        pred_masks: Optional[Tensor] = None,
        ood_scores: Optional[Tensor] = None,
    ) -> None: ...

    @abstractmethod
    def compute(
        self,
    ) -> MetricResults: ...

    @abstractmethod
    def reset(self) -> None: ...


class MeanIoU(Metric):
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        per_class: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.per_class = per_class
        self.device = resolve_device(device)
        self.reset()

    @torch.no_grad()
    def update(
        self,
        gt_masks: Tensor,
        pred_masks: Optional[Tensor] = None,
        ood_scores: Optional[Tensor] = None,
    ) -> None:
        if pred_masks is None:
            raise ValueError("MeanIoU requires 'pred_masks' argument.")

        if self.ignore_index is not None:
            valid_mask = gt_masks != self.ignore_index
        else:
            valid_mask = torch.ones_like(gt_masks, dtype=torch.bool)

        for cls in range(self.num_classes):
            pred_mask = (pred_masks == cls) & valid_mask
            gt_mask = (gt_masks == cls) & valid_mask
            self.intersection[cls] += torch.sum(pred_mask & gt_mask).float()
            self.union[cls] += torch.sum(pred_mask | gt_mask).float()

    def compute(self) -> MetricResults:
        iou = self.intersection / (self.union + 1e-6)  # Avoid division by zero
        valid_mask = self.union > 0  # Ignore classes with no samples

        miou = float(torch.mean(iou[valid_mask])) if torch.any(valid_mask) else 0.0

        result = {"mIoU": miou}

        if self.per_class:
            for cls in range(self.num_classes):
                if valid_mask[cls]:
                    result[f"IoU_class_{cls}"] = float(iou[cls])

        return result

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes, device=self.device)
        self.union = torch.zeros(self.num_classes, device=self.device)


class OoDAUPR(Metric):
    def __init__(
        self,
        unknown_label: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.unknown_label = unknown_label
        self.device = resolve_device(device)
        self.reset()

    @torch.no_grad()
    def update(
        self,
        gt_masks: Tensor,
        pred_masks: Optional[Tensor] = None,
        ood_scores: Optional[Tensor] = None,
    ) -> None:
        if ood_scores is None:
            raise ValueError("OoDAUPR requires 'ood_scores' argument.")

        ood_gt_mask = (gt_masks == self.unknown_label).long()

        B = ood_scores.shape[0]
        for i in range(B):
            ood_scores_flat = ood_scores[i].view(-1)
            ood_gt_mask_flat = ood_gt_mask[i].view(-1)

            p, r, _ = binary_precision_recall_curve(ood_scores_flat, ood_gt_mask_flat)
            self.aupr_out += auc(r, p)
            self.count += 1

    def compute(self) -> MetricResults:
        if self.count == 0:
            return {"OoDAUPR": 0.0}
        return {"OoDAUPR": float(self.aupr_out / self.count)}

    def reset(self) -> None:
        self.aupr_out = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.count = 0


class MetricCollection(Metric):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    @torch.no_grad()
    def update(
        self,
        gt_masks: Tensor,
        pred_masks: Optional[Tensor] = None,
        ood_scores: Optional[Tensor] = None,
    ):
        for metric in self.metrics:
            metric.update(gt_masks, pred_masks, ood_scores)

    def compute(self) -> MetricResults:
        combined = {}

        for metric in self.metrics:
            metric_dict = metric.compute()
            for k, v in metric_dict.items():
                combined[k] = v

        return combined

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
