import torch

from ml4cv_assignment.models.layers import FAM


def test_fam_block():
    batch_size = 2
    c1, c2 = 1024, 128
    h1, w1 = 32, 32
    h2, w2 = 16, 16

    feat_l = torch.rand(batch_size, c1, h1, w1)
    feat_s = torch.rand(batch_size, c2, h2, w2)

    fam = FAM(c1, c2)

    out = fam(feat_l, feat_s)

    print("-" * 30)
    print(f"Input feat_l: {feat_l.shape}")
    print(f"Input feat_s: {feat_s.shape}")
    print(f"Output      : {out.shape}")
    print("-" * 30)

    expected_shape = (batch_size, c2, h1, w1)
    assert out.shape == expected_shape, (
        f"Expected output shape {expected_shape} but got {out.shape}"
    )
