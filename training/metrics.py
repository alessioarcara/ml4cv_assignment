import abc

import torch
from loguru import logger
from torchmetrics.functional.classification import binary_precision_recall_curve
from torchmetrics.utilities.compute import auc


class Metric(abc.ABC):
    @abc.abstractmethod
    def update(self, logits: torch.Tensor, pred: torch.Tensor, true: torch.Tensor):
        pass

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class MeanIoU(Metric):
    def __init__(self, num_classes: int, device=None):
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    @torch.no_grad()
    def update(self, _: torch.Tensor, pred: torch.Tensor, true: torch.Tensor):
        for cls in range(self.num_classes):
            pred_mask = pred == cls
            true_mask = true == cls

            self.intersection[cls] += torch.sum(pred_mask & true_mask).float()
            self.union[cls] += torch.sum(pred_mask | true_mask).float()

    def compute(self):
        iou = self.intersection / (self.union + 1e-6)  # Avoid division by zero
        valid_iou = iou[self.union > 0]  # Ignore classes with no samples
        return (
            torch.mean(valid_iou)
            if len(valid_iou) > 0
            else torch.tensor(0.0, device=self.device)
        )

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, device=self.device)
        self.union = torch.zeros(self.num_classes, device=self.device)


class AUPR(Metric):
    def __init__(self, unknown_label: int, device=None):
        self.unknown_label = unknown_label
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, _: torch.Tensor, true: torch.Tensor):
        true = (true == self.unknown_label).long()

        # Sanity checks
        assert logits.shape == true.shape
        assert len(logits.shape) == 3
        assert len(true.shape) == 3
        assert logits.device == true.device

        for i in range(logits.shape[0]):
            logits_flat = logits[i].view(-1)
            true_flat = true[i].view(-1)

            p, r, _ = binary_precision_recall_curve(logits_flat, true_flat)
            self.aupr_out += auc(r, p)
            self.count += 1

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.device)
        return self.aupr_out / self.count

    def reset(self):
        self.aupr_out = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(0, dtype=torch.int64, device=self.device)


if __name__ == "__main__":
    mean_iou = MeanIoU(num_classes=10)
    aupr = AUPR(unknown_label=255)

    pred = torch.randint(0, 10, (4, 128, 128))
    true = torch.randint(0, 10, (4, 128, 128))
    mean_iou.update(None, pred, true)
    logger.debug(f"Mean IoU: {mean_iou.compute()}")

    logits = torch.randn(4, 128, 128)
    true[true == 4] = 255  # Set some pixels to unknown
    aupr.update(logits, None, true)
    logger.debug(f"AUPR: {aupr.compute()}")
