import torch

from ml4cv_assignment.training.metrics import AUPR, MeanIoU, MetricCollection


def test_metrics():
    metrics = MetricCollection(
        metrics=[
            MeanIoU(num_classes=10, ignore_index=255, per_class=True),
            AUPR(unknown_label=255),
        ]
    )

    B, H, W, C = 4, 128, 128, 10
    logits = torch.randn(B, H, W, C)
    pred = logits.argmax(-1)
    true = torch.randint(0, C, (B, H, W))
    true[true == 4] = 255  # Set some pixels to unknown

    ood_logits = logits[..., 4]

    metrics.update(ood_logits, pred, true)

    metric_results = metrics.compute()

    assert isinstance(metric_results, dict), (
        "MetricCollection.compute() should return a dictionary"
    )
    assert "mIoU" in metric_results
    assert "AUPR" in metric_results
