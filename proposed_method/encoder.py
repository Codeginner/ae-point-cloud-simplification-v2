"""
encoder.py — DGCNN encoder, faithful port of Yue Wang's implementation.

CHANGES vs original:
    • Restored EdgeConv4 (256-dim) yang di-comment out sebelumnya
    • Restored conv5 fusion MLP: concat(64+128+256+256)=704 → 1024
      Ini critical — tanpa conv5, per-point features jauh lebih noisy
      dan cls head tidak punya representasi yang cukup kuat.
    • Output dim: 1024 (bukan 448)
    • Semua downstream yang consume f_i perlu disesuaikan (scorer, cls_head, decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def knn(x: Tensor, k: int) -> Tensor:
    inner            = -2 * torch.matmul(x.transpose(2, 1), x)
    xx               = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx              = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x: Tensor, k: int = 20, idx: Tensor = None) -> Tensor:
    batch_size = x.size(0)
    num_points = x.size(2)
    x          = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)

    device   = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx      = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()

    x       = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x       = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: Tensor, k: int, idx: Tensor = None) -> Tensor:
        feat = get_graph_feature(x, k=k, idx=idx)
        out  = self.mlp(feat)
        return out.max(dim=-1, keepdim=False)[0]


class DGCNNEncoder(nn.Module):
    """
    Full DGCNN encoder — 4 EdgeConv layers + fusion MLP.

    Architecture (APES-equivalent):
        EdgeConv1 : 6   → 64
        EdgeConv2 : 128 → 64
        EdgeConv3 : 128 → 128
        EdgeConv4 : 256 → 256
        conv5     : concat(64+64+128+256)=512 → 1024  [fusion MLP]

    Output: per-point features f_i ∈ R^1024
    """

    # Feature dim exposed so downstream modules can read it
    out_dim: int = 1024

    def __init__(self, k: int = 20) -> None:
        super().__init__()
        self.k = k

        self.conv1 = nn.Sequential(
            nn.Conv2d(6,      64,  kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64,  kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        # Fusion MLP: concat(64+64+128+256)=512 → 1024
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4])

    def forward(self, P: Tensor) -> Tensor:
        """
        Args:
            P: Input point cloud (B, N, 3).

        Returns:
            f_i: Per-point features (B, N, 1024).
        """
        x = P.permute(0, 2, 1)   # (B, 3, N)

        feat = get_graph_feature(x, k=self.k)
        x1   = self.conv1(feat).max(dim=-1, keepdim=False)[0]   # (B, 64, N)

        feat = get_graph_feature(x1, k=self.k)
        x2   = self.conv2(feat).max(dim=-1, keepdim=False)[0]   # (B, 64, N)

        feat = get_graph_feature(x2, k=self.k)
        x3   = self.conv3(feat).max(dim=-1, keepdim=False)[0]   # (B, 128, N)

        feat = get_graph_feature(x3, k=self.k)
        x4   = self.conv4(feat).max(dim=-1, keepdim=False)[0]   # (B, 256, N)

        # Fusion
        x   = torch.cat((x1, x2, x3, x4), dim=1)               # (B, 512, N)
        x   = self.conv5(x)                                      # (B, 1024, N)

        f_i = x.permute(0, 2, 1)                                 # (B, N, 1024)
        return f_i
