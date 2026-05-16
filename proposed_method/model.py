"""
model.py — PointCloudSimplifier: main orchestrator.

Full forward pipeline:

    P (B,N,3)
    ├── DGCNNEncoder          → f_i   (B, N, 448)
    ├── NCScoreModule         → s_i   (B, N)
    ├── ImportanceScoringMLP  → score (B, N)
    ├── AdaptiveSelector      → idx   (B, M)
    │    └── gather P, f_i → P_s (B,M,3), f_s (B,M,448)
    ├── FoldingNetDecoder     → P_recon (B, M, 3)
    └── GeometryAwareLoss     → loss_dict

Class:
    PointCloudSimplifier
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
# Lightweight Classification Head
# Takes simplified point features f_s (B, M, 448) → class logits (B, C)
# Trained jointly with reconstruction — gradient flows through STE to selector
# ===========================================================================

class ClsHead(nn.Module):
    """
    Classification head over simplified point features.

    Uses global max+mean pooling over f_s then a 2-layer MLP.
    Deliberately lightweight — the heavy lifting is done by the DGCNN encoder.

    Args:
        in_dim    : feature dimension from encoder (448)
        num_class : number of output classes
        dropout   : dropout rate
    """
    def __init__(self, in_dim: int = 448, num_class: int = 10, dropout: float = 0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            # Project pooled features
            nn.Linear(in_dim * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_class),
        )

    def forward(self, f_s: Tensor) -> Tensor:
        """
        Args:
            f_s : simplified point features (B, M, 448)
        Returns:
            logits : (B, num_class)
        """
        # Global descriptor: max + mean pooling over M simplified points
        g = torch.cat([f_s.max(dim=1).values,
                       f_s.mean(dim=1)], dim=-1)   # (B, 896)
        return self.mlp(g)                          # (B, num_class)


class PointCloudSimplifier(nn.Module):
    """
    End-to-end point cloud simplification framework.

    Proposed-method pipeline:

        Point Cloud
            ↓
        DGCNN Encoder
            ↓
        NC Score
            ↓
        Importance Scoring
            ↓
        Geometry-Balanced Selection
            ↓
        FoldingNet Reconstruction
    """

    def __init__(
        self,
        M: int = 1024,
        k: int = 20,
        alpha: float = 0.7,
        threshold: float = 0.5,
        latent_dim: int = 1024,
        lambda_1: float = 1.0,
        lambda_2: float = 0.5,
        lambda_3: float = 0.3,
        lambda_4: float = 0.5,
        num_class: int = 10,
        lambda_cls: float = 0.5,
    ) -> None:

        super().__init__()

        self.M = M
        self.lambda_cls = lambda_cls

        # --------------------------------------------------------------
        # Encoder
        # Output:
        # f_i shape = (B,N,448)
        # --------------------------------------------------------------

        self.encoder = DGCNNEncoder(k=k)

        # --------------------------------------------------------------
        # NC Score Module
        # No learnable parameters
        # --------------------------------------------------------------

        self.nc_module = NCScoreModule(k=k)

        for p in self.nc_module.parameters():
            p.requires_grad_(False)

        # --------------------------------------------------------------
        # Importance Scoring MLP
        # Input:
        # 448 + 1 = 449
        # --------------------------------------------------------------

        # ini bagian yang diubah
        self.scorer = ImportanceScoringMLP(
            in_dim=449,
        )

        # --------------------------------------------------------------
        # Adaptive Geometry-Balanced Selector
        # --------------------------------------------------------------

        self.selector = AdaptiveSelector(
            M=M,
            alpha=alpha,
            threshold=threshold,
        )

        # --------------------------------------------------------------
        # FoldingNet Decoder
        # Input feature:
        # 448 dim
        # --------------------------------------------------------------

        # ini bagian yang diubah
        self.decoder = FoldingNetDecoder(
            M=M,
            in_dim=448,
            latent_dim=latent_dim,
        )

        # --------------------------------------------------------------
        # Geometry-aware loss
        # --------------------------------------------------------------

        self.loss_fn = GeometryAwareLoss(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
        )

        # --------------------------------------------------------------
        # Classification Head (joint training)
        # Input: f_s (B, M, 448) → logits (B, num_class)
        # --------------------------------------------------------------

        self.cls_head = ClsHead(
            in_dim=448,
            num_class=num_class,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P: Tensor,
        labels: Tensor = None,
        compute_loss: bool = True,
    ) -> dict[str, Tensor]:
        """
        Args:
            P:
                Input point cloud
                shape = (B,N,3)

            compute_loss:
                Whether to compute geometry-aware loss.

        Returns:
            Dictionary containing:
                • simplified point cloud
                • reconstructed point cloud
                • importance scores
                • selected indices
                • optional loss dictionary
        """

        # --------------------------------------------------------------
        # 1. DGCNN Encoder
        # f_i shape = (B,N,448)
        # --------------------------------------------------------------

        f_i = self.encoder(P)

        # --------------------------------------------------------------
        # 2. NC Score
        # s_i shape = (B,N)
        # --------------------------------------------------------------

        with torch.no_grad():

            s_i = self.nc_module(P)

        # --------------------------------------------------------------
        # 3. Importance Scoring
        # score shape = (B,N)
        # --------------------------------------------------------------

        score = self.scorer(
            f_i,
            s_i
        )

        # --------------------------------------------------------------
        # 4. Adaptive Geometry-Balanced Selection
        # idx shape = (B,M)
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # 4. Adaptive Geometry-Balanced Selection  (vectorised + STE)
        #    Returns idx  (B, M)   — integer indices for feature gather
        #            P_s  (B, M, 3) — differentiable via STE
        # --------------------------------------------------------------

        idx, P_s = self.selector(P, score, s_i)

        # --------------------------------------------------------------
        # 5. Gather simplified features
        # --------------------------------------------------------------

        f_s = index_points(f_i, idx)            # (B, M, 448)

        # --------------------------------------------------------------
        # 6. FoldingNet Reconstruction
        # --------------------------------------------------------------

        P_recon = self.decoder(
            P_s,
            f_s
        )                                           # (B,M,3)

        # --------------------------------------------------------------
        # Output dictionary
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # 7. Classification Head
        # logits shape = (B, num_class)
        # --------------------------------------------------------------

        logits = self.cls_head(f_s)

        out = {
            "P_simplified": P_s,
            "P_recon":      P_recon,
            "score":        score,
            "idx":          idx,
            "logits":       logits,
        }

        # --------------------------------------------------------------
        # 8. Joint loss: geometry + classification
        # --------------------------------------------------------------

        if compute_loss:

            loss_dict = self.loss_fn(
                P_recon,
                P,
                P_s,
                score,
            )

            # Classification loss (only when labels provided)
            if labels is not None:
                L_cls = F.cross_entropy(logits, labels)
                loss_dict["cls"]   = L_cls
                loss_dict["total"] = loss_dict["total"] + self.lambda_cls * L_cls
            else:
                loss_dict["cls"] = torch.tensor(0.0, device=P.device)

            out["loss"] = loss_dict

        return out