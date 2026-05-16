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
from torch import Tensor

from .encoder import DGCNNEncoder
from .nc_score import NCScoreModule
from .scoring import ImportanceScoringMLP
from .selector import AdaptiveSelector
from .decoder import FoldingNetDecoder
from .loss import GeometryAwareLoss
from .utils import index_points


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
    ) -> None:

        super().__init__()

        self.M = M

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P: Tensor,
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

        out = {

            "P_simplified": P_s,

            "P_recon": P_recon,

            "score": score,

            "idx": idx,
        }

        # --------------------------------------------------------------
        # 7. Geometry-aware loss
        # --------------------------------------------------------------

        if compute_loss:

            loss_dict = self.loss_fn(
                P_recon,
                P,
                P_s,
                score,
            )

            out["loss"] = loss_dict

        return out