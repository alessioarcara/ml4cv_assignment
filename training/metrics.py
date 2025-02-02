import torch
import abc


class Metric(abc.ABC):
    @abc.abstractmethod
    def update(self, pred, true):
        pass

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class MeanIoU(Metric):
    def __init__(self, num_classes, device=None):
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def update(self, pred, true):
        for cls in range(self.num_classes):
            pred_mask = (pred == cls)
            true_mask = (true == cls)

            self.intersection[cls] += torch.sum(pred_mask & true_mask).float()
            self.union[cls] += torch.sum(pred_mask | true_mask).float()

    def compute(self):
        iou = self.intersection / (self.union + 1e-6) # Avoid division by zero
        valid_iou = iou[self.union > 0] # Ignore classes with no samples
        return torch.mean(valid_iou) if len(valid_iou) > 0 else torch.tensor(0.0, device=self.device)
    
    def reset(self):
        self.intersection = torch.zeros(self.num_classes, device=self.device)
        self.union = torch.zeros(self.num_classes, device=self.device)