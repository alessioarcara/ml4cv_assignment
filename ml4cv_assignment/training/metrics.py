from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_precision_recall_curve
from torchmetrics.utilities.compute import auc


class Metric(ABC):
    @abstractmethod
    def update(self, logits: Tensor, pred: Tensor, true: Tensor) -> None: ...

    @abstractmethod
    def compute(
        self,
    ) -> Union[float, List[float], Dict[str, Union[float, List[float]]]]: ...

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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reset()

    @torch.no_grad()
    def update(self, logits: Tensor, pred: Tensor, true: Tensor) -> None:
        if self.ignore_index is not None:
            valid_mask = true != self.ignore_index
        else:
            valid_mask = torch.ones_like(true, dtype=torch.bool)

        for cls in range(self.num_classes):
            pred_mask = (pred == cls) & valid_mask
            true_mask = (true == cls) & valid_mask

            self.intersection[cls] += torch.sum(pred_mask & true_mask).float()
            self.union[cls] += torch.sum(pred_mask | true_mask).float()

    def compute(self) -> Union[float, List[float]]:
        iou = self.intersection / (self.union + 1e-6)  # Avoid division by zero
        valid_iou = iou[self.union > 0]  # Ignore classes with no samples

        if self.per_class:
            # Return IoU for each class, with 0 for classes with no samples
            return iou.tolist()
        else:
            # Return mean IoU over classes with at least one sample
            return float(torch.mean(valid_iou)) if len(valid_iou) > 0 else 0.0

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes, device=self.device)
        self.union = torch.zeros(self.num_classes, device=self.device)


class AUPR(Metric):
    def __init__(
        self,
        unknown_label: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.unknown_label = unknown_label
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reset()

    @torch.no_grad()
    def update(self, logits: Tensor, pred: Tensor, true: Tensor) -> None:
        true = (true == self.unknown_label).long()

        # Sanity checks
        assert logits.shape == true.shape
        assert len(logits.shape) == 3
        assert len(true.shape) == 3
        assert logits.device == true.device

        B = logits.shape[0]
        for i in range(B):
            logits_flat = logits[i].view(-1)
            true_flat = true[i].view(-1)

            p, r, _ = binary_precision_recall_curve(logits_flat, true_flat)
            self.aupr_out += auc(r, p)
            self.count += 1

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.aupr_out / self.count)

    def reset(self) -> None:
        self.aupr_out = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.count = 0


class MetricCollection(Metric):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    @torch.no_grad()
    def update(self, logits: Tensor, pred: Tensor, true: Tensor):
        for metric in self.metrics:
            metric.update(logits, pred, true)

    def compute(self) -> Dict[str, Union[float, List[float]]]:
        return {metric.__class__.__name__: metric.compute() for metric in self.metrics}  # type: ignore

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
