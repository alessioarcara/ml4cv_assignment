import torch

from ml4cv_assignment.training.losses import RejectedByAllLoss


def test_rejected_by_all_loss():
    criterion = RejectedByAllLoss()
    # 1 Pixel, labeled as 13 (OOD)
    # Shape: (1, 1, 1)
    target = torch.tensor([[[13]]])

    # Case 1: The model is uncertain
    logits_zero = torch.zeros(1, 2, 1, 1, requires_grad=True)
    loss1 = criterion(logits_zero, target)

    expected_loss1 = torch.tensor(25.0)

    assert torch.isclose(loss1, expected_loss1), (
        f"Case 1 Failed: {expected_loss1.item()}, but got: {loss1.item()}"
    )

    # Case 2: The model is CONFIDENT that the pixel is Class 0 (Logit=10),
    # and is uncertain about Class 1 (Logit=0).
    logits_confident = torch.tensor([10.0, 0.0]).view(1, 2, 1, 1)
    logits_confident.requires_grad = True
    loss2 = criterion(logits_confident, target)

    # Math Logic:
    # Tanh(10) ≈ 1.0
    # Tanh(0)  = 0.0
    # Sum      = 1.0
    # Neg Sum  = -1.0 (because sum_probs = -probs.sum())
    # Loss     = (alpha - 1.0)^2 = (5.0 - 1.0)^2 = 16.0
    expected_loss2 = torch.tensor(16.0)

    assert torch.isclose(loss2, expected_loss2, atol=1e-3), (
        f"Case 2 Failed. Expected {expected_loss2.item()}, but got: {loss2.item()}"
    )
