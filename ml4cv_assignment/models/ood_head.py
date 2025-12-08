import torch.nn as nn
from torch import Tensor


class OoDHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        use_batch_norm: bool,
        use_leaky_relu: bool,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=not use_batch_norm),
            nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=not use_batch_norm),
            nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
