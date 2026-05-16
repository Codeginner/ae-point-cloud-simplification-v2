"""
utils.py — Shared utility functions for proposed_method.

Functions:
    knn_graph      : build k-nearest-neighbour index tensor
    normalize_pc   : zero-mean + unit-sphere normalisation
    fps            : farthest point sampling
"""

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# KNN graph
# ---------------------------------------------------------------------------

def knn_graph(x: Tensor, k: int) -> Tensor:
    """Return indices of k nearest neighbours for every point.

    Args:
        x: Point cloud of shape (B, N, C).
        k: Number of neighbours.

    Returns:
        idx: LongTensor of shape (B, N, k) — neighbour indices.
    """
    # Pairwise squared distances
    # x: (B, N, C)
    B, N, C = x.shape
    inner = -2.0 * torch.bmm(x, x.transpose(2, 1))   # (B, N, N)
    xx    = (x ** 2).sum(dim=-1, keepdim=True)         # (B, N, 1)
    dist  = xx + inner + xx.transpose(2, 1)            # (B, N, N)

    # k+1 because the closest point to x_i is x_i itself
    _, idx = dist.topk(k=k + 1, dim=-1, largest=False)  # (B, N, k+1)
    idx = idx[:, :, 1:]                                  # drop self; (B, N, k)
    return idx


# ---------------------------------------------------------------------------
# Point cloud normalisation
# ---------------------------------------------------------------------------

def normalize_pc(x: Tensor) -> Tensor:
    """Translate to zero mean and scale into unit sphere.

    Args:
        x: (B, N, 3) or (N, 3).

    Returns:
        x_norm: Same shape as input.
    """
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)

    centroid = x.mean(dim=1, keepdim=True)          # (B, 1, 3)
    x = x - centroid
    scale = x.norm(dim=-1).max(dim=1).values        # (B,)
    x = x / (scale[:, None, None] + 1e-8)

    if not batched:
        x = x.squeeze(0)
    return x


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def fps(x: Tensor, n_samples: int) -> Tensor:
    """Farthest point sampling.

    Args:
        x:         (B, N, 3)
        n_samples: Number of points to sample M.

    Returns:
        idx: (B, M) — sampled point indices.
    """
    B, N, _ = x.shape
    device  = x.device

    sampled_idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances   = torch.full((B, N), float("inf"), device=device)

    # Start from a random point in each batch
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_samples):
        sampled_idx[:, i] = farthest
        # Gather current farthest point: (B, 1, 3)
        cur = x[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = ((x - cur) ** 2).sum(dim=-1)            # (B, N)
        distances = torch.minimum(distances, dist)
        farthest  = distances.argmax(dim=-1)            # (B,)

    return sampled_idx


# ---------------------------------------------------------------------------
# Gather points by index
# ---------------------------------------------------------------------------

def index_points(x: Tensor, idx: Tensor) -> Tensor:
    """Gather points from x using idx.

    Args:
        x:   (B, N, C)
        idx: (B, M) or (B, N, k)

    Returns:
        out: (B, M, C) or (B, N, k, C)
    """
    B = x.shape[0]
    flat_idx = idx.reshape(B, -1)                       # (B, M*k)
    out = x[torch.arange(B, device=x.device).unsqueeze(-1), flat_idx]
    return out.reshape(*idx.shape, x.shape[-1])
