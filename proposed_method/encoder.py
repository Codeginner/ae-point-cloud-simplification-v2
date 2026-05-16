"""
encoder.py — DGCNN encoder, faithful port of Yue Wang's implementation.

Reference:
    Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", 2019.
    Original: https://github.com/WangYueFt/dgcnn  (model.py, class get_model)

Key differences from the original:
    • The original is a full classification network (EdgeConv × 4 + global pool
      + MLP head). Here we keep only the **per-point feature extractor** part
      (EdgeConv × 4 + conv5 fusion) and return per-point features f_i instead
      of a class logit, so the output can be fed into ImportanceScoringMLP.
    • Input expected as (B, N, 3) matching the rest of this codebase;
      internally transposed to (B, 3, N) as in the original.
    • `get_graph_feature` is kept verbatim except the hard-coded
      `device = torch.device('cuda')` is replaced with `x.device` so the
      module works on CPU too.

Classes:
    EdgeConvLayer  : thin wrapper exposing one EdgeConv block (for the OOP diagram)
    DGCNNEncoder   : 4-layer DGCNN producing per-point features f_i ∈ R^emb_dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Verbatim helpers from Yue Wang's model.py
# ---------------------------------------------------------------------------

def knn(x: Tensor, k: int) -> Tensor:
    """Return indices of k nearest neighbours.

    Identical to Wang et al. original except device is inferred from x.

    Args:
        x: (B, C, N)  — note: channels-first, as in the original.
        k: Number of neighbours.

    Returns:
        idx: (B, N, k)
    """
    inner            = -2 * torch.matmul(x.transpose(2, 1), x)   # (B, N, N)
    xx               = torch.sum(x ** 2, dim=1, keepdim=True)     # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)          # (B, N, N)
    idx              = pairwise_distance.topk(k=k, dim=-1)[1]     # (B, N, k)
    return idx


def get_graph_feature(x: Tensor, k: int = 20, idx: Tensor = None) -> Tensor:
    """Build edge feature tensor [x_j − x_i || x_i] for all neighbours.

    Identical to Wang et al. original except `device` is inferred from x
    instead of being hard-coded to 'cuda'.

    Args:
        x:   (B, C, N)
        k:   Number of neighbours.
        idx: Pre-computed KNN indices (B, N, k). Computed if None.

    Returns:
        feature: (B, 2C, N, k)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x          = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)                                          # (B, N, k)

    device   = x.device                                            # ← only change vs. original
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx      = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()

    x       = x.transpose(2, 1).contiguous()                      # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x       = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature                                                  # (B, 2C, N, k)


# ---------------------------------------------------------------------------
# EdgeConvLayer  — thin OOP wrapper around one conv block
# ---------------------------------------------------------------------------

class EdgeConvLayer(nn.Module):
    """One EdgeConv block: Conv2d(2C → C_out) + BN + LeakyReLU + max-pool.

    Wraps the pattern used repeatedly in Wang et al.'s forward():
        x = get_graph_feature(x_prev, k)
        x = conv(x)          # (B, C_out, N, k)
        x = x.max(dim=-1)[0] # (B, C_out, N)

    Args:
        in_channels:  C_in  (edge feature dim = 2 * prev output channels).
        out_channels: C_out.
    """

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
        """
        Args:
            x:   (B, C_in // 2, N)  — output of previous layer (not yet doubled).
            k:   Number of neighbours.
            idx: Optional pre-computed KNN indices.

        Returns:
            out: (B, C_out, N)
        """
        feat = get_graph_feature(x, k=k, idx=idx)   # (B, C_in, N, k)
        out  = self.mlp(feat)                         # (B, C_out, N, k)
        return out.max(dim=-1, keepdim=False)[0]      # (B, C_out, N)


# ---------------------------------------------------------------------------
# DGCNNEncoder  — 4-layer encoder, per-point features
# ---------------------------------------------------------------------------

class DGCNNEncoder(nn.Module):
    """
    Proposed-method DGCNN encoder.

    Architecture:
        EdgeConv1 : 64
        EdgeConv2 : 128
        EdgeConv3 : 256

    Final feature:
        concat(x1,x2,x3) -> 448 dim
    """

    def __init__(self, k: int = 20) -> None: #changed line
        super().__init__()
        self.k        = k
        # self.emb_dims = emb_dims

        # EdgeConv blocks — in_channels = 2 × prev_out (edge feature concat)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6,     64,  kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        #self.conv2 = nn.Sequential(
        #    nn.Conv2d(64*2,  64,  kernel_size=1, bias=False),
        #    nn.BatchNorm2d(64),
        #    nn.LeakyReLU(negative_slope=0.2),
        #)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        #self.conv3 = nn.Sequential(
        #    nn.Conv2d(64*2,  128, kernel_size=1, bias=False),
        #    nn.BatchNorm2d(128),
        #    nn.LeakyReLU(negative_slope=0.2),
        #)
        #self.conv4 = nn.Sequential(
        #    nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
        #    nn.BatchNorm2d(256),
        #    nn.LeakyReLU(negative_slope=0.2),
        #)

        # Fusion: concat(x1,x2,x3,x4) = 64+64+128+256 = 512 → emb_dims
        #self.conv5 = nn.Sequential(
        #    nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
        #    nn.BatchNorm1d(emb_dims),
        #    nn.LeakyReLU(negative_slope=0.2),
        #)

        # Expose EdgeConv blocks via ModuleList for the OOP diagram
        
        #self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4])

        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3])

    def forward(self, P: Tensor) -> Tensor:
        """
        Args:
            P: Input point cloud (B, N, 3).

        Returns:
            f_i: Per-point features (B, N, emb_dims).
        """
        x = P.permute(0, 2, 1)                            # (B, 3, N)

        # EdgeConv block 1  — input: raw xyz (3-dim), edge feat: 6
        feat = get_graph_feature(x, k=self.k)             # (B, 6, N, k)
        x1   = self.conv1(feat).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # EdgeConv block 2
        feat = get_graph_feature(x1, k=self.k)            # (B, 128, N, k)
        x2   = self.conv2(feat).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # EdgeConv block 3
        feat = get_graph_feature(x2, k=self.k)            # (B, 128, N, k)
        x3   = self.conv3(feat).max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        # EdgeConv block 4
        # feat = get_graph_feature(x3, k=self.k)            # (B, 256, N, k)
        # x4   = self.conv4(feat).max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        # Multi-scale concat + fusion
        x = torch.cat((x1, x2, x3), dim=1)
        # x    = torch.cat((x1, x2, x3, x4), dim=1)        # (B, 512, N)
        # x    = self.conv5(x)                               # (B, emb_dims, N)

        f_i  = x.permute(0, 2, 1)                         # (B, N, emb_dims)
        return f_i
