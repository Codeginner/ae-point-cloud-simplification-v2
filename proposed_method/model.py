"""
model.py — PointCloudSimplifier: main orchestrator.

CHANGES vs original:
    • Encoder output dim: 1024 (restored full DGCNN)
    • Scorer input: 1024+1 = 1025
    • Decoder in_dim: 1024
    • ClsHead in_dim: 1024
    • lambda_cls default: 1.0 (naik dari 0.5)
    • lambda_4 default: 0.1 (turun — score supervision terlalu dominan)
    • ClsHead: hapus self-attention (mahal + overfitting pada simplified features)
      ganti ke DGCNN-style EdgeConv head yang lebih cocok untuk point sets
    • alpha selector: 0.6 (turun dari 0.7) — beri lebih banyak flat points
      karena untuk klasifikasi global shape, flat regions juga penting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder import DGCNNEncoder
from .nc_score import NCScoreModule
from .scoring import ImportanceScoringMLP
from .selector import AdaptiveSelector
from .decoder import FoldingNetDecoder
from .loss import GeometryAwareLoss
from .utils import index_points


# ===========================================================================
# Classification Head — DGCNN-style, no self-attention
# Lebih cocok untuk per-point features dari simplified cloud
# ===========================================================================

def knn_head(x: Tensor, k: int) -> Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx    = (x ** 2).sum(dim=1, keepdim=True)
    dist  = -xx - inner - xx.transpose(2, 1)
    return dist.topk(k, dim=-1)[1]


def get_edge_feat(x: Tensor, k: int) -> Tensor:
    B, C, N = x.shape
    idx     = knn_head(x, k)
    device  = x.device
    base    = torch.arange(B, device=device).view(-1, 1, 1) * N
    flat    = (idx + base).view(-1)
    x_t     = x.transpose(2, 1).contiguous().view(B * N, C)
    neigh   = x_t[flat].view(B, N, k, C).permute(0, 3, 1, 2)
    xi      = x.unsqueeze(-1).expand_as(neigh)
    return torch.cat([xi, neigh - xi], dim=1)   # (B, 2C, N, k)


class ClsHead(nn.Module):
    """
    DGCNN-style classification head.

    Input : f_s (B, M, in_dim)  — simplified point features
    Output: logits (B, num_class)

    Architecture:
        EdgeConv(in_dim → 256) + max-pool
        EdgeConv(256 → 256) + max-pool
        concat max+mean global → 512
        Linear(512 → 256 → num_class)

    Kenapa bukan self-attention:
        Self-attention bagus untuk long-range context, tapi untuk M=512
        points yang sudah di-simplify, EdgeConv yang pakai local geometry
        lebih robust — terutama karena simplified cloud bisa punya distribusi
        yang irregular (clustered di contours).
    """

    def __init__(self, in_dim: int = 1024, num_class: int = 10,
                 k: int = 20, dropout: float = 0.5):
        super().__init__()
        self.k = k

        # Project dulu ke 256 supaya EdgeConv tidak terlalu mahal
        self.proj = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        self.ec1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.ec2 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_class),
        )

    def forward(self, f_s: Tensor) -> Tensor:
        """f_s: (B, M, in_dim) → logits: (B, num_class)"""
        x = f_s.permute(0, 2, 1)           # (B, in_dim, M)
        x = self.proj(x)                    # (B, 256, M)

        k = min(self.k, x.shape[-1] - 1)

        e1 = self.ec1(get_edge_feat(x, k)).max(dim=-1)[0]   # (B, 256, M)
        e2 = self.ec2(get_edge_feat(e1, k)).max(dim=-1)[0]  # (B, 256, M)

        g = torch.cat([e1.max(-1)[0], e2.max(-1)[0]], dim=1)  # (B, 512)
        return self.mlp(g)


class PointCloudSimplifier(nn.Module):
    """
    End-to-end point cloud simplification framework.
    """

    def __init__(
        self,
        M: int = 512,
        k: int = 20,
        alpha: float = 0.6,          # CHANGED: 0.7→0.6
        threshold: float = 0.5,
        latent_dim: int = 1024,
        lambda_1: float = 1.0,
        lambda_2: float = 0.5,
        lambda_3: float = 0.3,
        lambda_4: float = 0.1,       # CHANGED: 0.3→0.1 (score supervision less dominant)
        num_class: int = 10,
        lambda_cls: float = 1.0,     # CHANGED: 0.5→1.0
    ) -> None:

        super().__init__()

        self.M = M
        self.lambda_cls = lambda_cls

        # Encoder — full 4-layer DGCNN, output dim = 1024
        self.encoder = DGCNNEncoder(k=k)
        enc_dim = self.encoder.out_dim   # 1024

        # NC Score — no learnable params
        self.nc_module = NCScoreModule(k=k)
        for p in self.nc_module.parameters():
            p.requires_grad_(False)

        # Scorer: enc_dim + 1 (nc score)
        self.scorer = ImportanceScoringMLP(in_dim=enc_dim + 1)

        # Selector
        self.selector = AdaptiveSelector(M=M, alpha=alpha, threshold=threshold)

        # Decoder: in_dim = enc_dim
        self.decoder = FoldingNetDecoder(M=M, in_dim=enc_dim, latent_dim=latent_dim)

        # Loss
        self.loss_fn = GeometryAwareLoss(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
        )

        # Cls Head: in_dim = enc_dim
        self.cls_head = ClsHead(in_dim=enc_dim, num_class=num_class, k=min(k, M - 1))

    def forward(
        self,
        P: Tensor,
        labels: Tensor = None,
        compute_loss: bool = True,
    ) -> dict:

        # 1. Encoder → (B, N, 1024)
        f_i = self.encoder(P)

        # 2. NC Score → (B, N)
        with torch.no_grad():
            s_i = self.nc_module(P)

        # 3. Importance Scoring → (B, N)
        score = self.scorer(f_i, s_i)

        # 4. Adaptive Selection → idx (B,M), P_s (B,M,3)
        idx, P_s = self.selector(P, score, s_i)

        # 5. Gather simplified features → (B, M, 1024)
        f_s = index_points(f_i, idx)

        # 6. Decode → (B, M, 3)
        P_recon = self.decoder(P_s, f_s)

        # 7. Classification → (B, num_class)
        logits = self.cls_head(f_s)

        out = {
            "P_simplified": P_s,
            "P_recon":      P_recon,
            "score":        score,
            "idx":          idx,
            "logits":       logits,
        }

        if compute_loss:
            loss_dict = self.loss_fn(P_recon, P, P_s, score)

            if labels is not None:
                L_cls = F.cross_entropy(logits, labels)
                loss_dict["cls"]   = L_cls
                loss_dict["total"] = loss_dict["total"] + self.lambda_cls * L_cls
            else:
                loss_dict["cls"] = torch.tensor(0.0, device=P.device)

            out["loss"] = loss_dict

        return out

# Alias for backwards-compatible imports
ProposedSimplifier = PointCloudSimplifier
