import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d

# Implementazione di FaPN basata su:
# https://github.com/sithu31296/semantic-segmentation

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, g=1):
        super().__init__()
        self.offset_mask = nn.Conv2d(in_channels, g * 3 * k * k, k, s, p)
        self.dcn = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=k, 
            stride=s, 
            padding=p, 
            groups=g, 
        )
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_atten = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_conv = FSM(in_channels, out_channels)
        self.offset = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
        self.dcpack_l2 = DCNv2(out_channels, out_channels, 3, 1, 1, 8)
    
    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm


if __name__ == '__main__':
    batch_size = 2
    c1, c2 = 1024, 128 
    h1, w1 = 32, 32
    h2, w2 = 16, 16

    feat_l = torch.rand(batch_size, c1, h1, w1)
    feat_s = torch.rand(batch_size, c2, h2, w2)

    fam = FAM(c1, c2)

    out = fam(feat_l, feat_s)
    print("Input feat_l shape:", feat_l.shape)
    print("Input feat_s shape:", feat_s.shape)
    print("Output shape:", out.shape)