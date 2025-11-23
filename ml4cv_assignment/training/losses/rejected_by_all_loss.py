import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RejectedByAllLoss(nn.Module):
    def __init__(self, alpha: float = 5.0, unknown_label: int = -1) -> None:
        super().__init__()
        self.alpha = alpha
        self.unknown_label = unknown_label

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        # Identify OOD pixels that belong to the outlier class
        ood_mask = targets == self.unknown_label

        if not ood_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute probabilities using tanh. Why tanh?
        # Sigmoid(0) = 0.5. If the model is uncertain across K classes, the sum of
        # probabilities would be K*0.5, creating a false "inlier" score.
        # Tanh(0) = 0, so uncertain predictions contribute 0 to the sum,
        # correctly identifying the pixel as having low energy (high OOD likelihood).
        probs = logits.tanh()

        # Negative Sum: lower accumulated probability = higher OOD likelihood
        sum_probs = probs.sum(dim=1)

        # Select sum_probs only for the ood pixels
        ood_sum_probs = sum_probs[ood_mask]

        # max(0, alpha + sum_probs)^2 (Squared Hinge Loss)
        # This penalizes the model if the 'ood_sum_probs' is greater than '-alpha'
        loss = torch.pow(F.relu(self.alpha + ood_sum_probs), 2).mean()

        return loss
