import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, loss: nn.Module, weight: float = 1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs) * self.weight
