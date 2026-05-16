"""
scoring.py — Importance Scoring MLP.

Fuses:
    • DGCNN multi-scale feature  f_i ∈ R^448
    • NC score                   s_i ∈ R^1

into:

    g_i = concat(f_i, s_i) ∈ R^449

Then predicts a scalar importance score:

    score_i ∈ [0,1]

Architecture:
    449 → 128 → 64 → 1 → Sigmoid

Class:
    ImportanceScoringMLP
"""

import torch
import torch.nn as nn
from torch import Tensor


class ImportanceScoringMLP(nn.Module):
    """
    Per-point importance scoring module.

    Proposed-method version:
        • Input:
            448-dim DGCNN feature
            +
            1-dim NC score

        • Output:
            scalar importance score ∈ [0,1]

    Args:
        in_dim:
            Input feature dimension.

            Proposed method:
                448 + 1 = 449

        hidden_dims:
            Hidden layer dimensions.

        dropout:
            Dropout probability.
    """

    def __init__(
        self,
        # ini bagian yang diubah
        in_dim: int = 449,

        hidden_dims: tuple = (128, 64),

        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim

        # --------------------------------------------------------------
        # MLP:
        # 449 → 128 → 64 → 1 → Sigmoid
        # --------------------------------------------------------------

        layers: list[nn.Module] = []

        prev = in_dim

        for h in hidden_dims:

            layers += [

                # Fully-connected layer
                nn.Linear(prev, h),

                # BatchNorm applied after flattening:
                # shape = (B*N, h)
                nn.BatchNorm1d(h),

                # Non-linearity
                nn.ReLU(inplace=True),

                # Regularization
                nn.Dropout(dropout),
            ]

            prev = h

        # Final scalar importance score
        layers += [

            nn.Linear(prev, 1),

            # Score ∈ [0,1]
            nn.Sigmoid()
        ]

        self.layers = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        f_i: Tensor,
        s_i: Tensor,
    ) -> Tensor:
        """
        Args:
            f_i:
                Per-point DGCNN feature
                shape = (B, N, 448)

            s_i:
                Per-point NC score
                shape = (B, N)

        Returns:
            score:
                Per-point importance score
                shape = (B, N)
        """

        B, N, _ = f_i.shape

        # --------------------------------------------------------------
        # Eq. (6)
        # Expand NC score:
        # (B,N) -> (B,N,1)
        # --------------------------------------------------------------

        # ini bagian yang diubah
        s_i_expanded = s_i.unsqueeze(-1)

        # --------------------------------------------------------------
        # Eq. (6)
        # g_i = concat(f_i, s_i)
        # Shape:
        # (B,N,448+1) = (B,N,449)
        # --------------------------------------------------------------

        # ini bagian yang diubah
        g_i = torch.cat(
            [f_i, s_i_expanded],
            dim=-1
        )

        # --------------------------------------------------------------
        # Flatten:
        # (B,N,449) -> (B*N,449)
        # Needed for Linear + BatchNorm1d
        # --------------------------------------------------------------

        # ini bagian yang diubah
        x = g_i.reshape(B * N, -1)

        # --------------------------------------------------------------
        # MLP scoring
        # --------------------------------------------------------------

        x = self.layers(x)

        # --------------------------------------------------------------
        # Restore point-cloud shape
        # (B*N,1) -> (B,N)
        # --------------------------------------------------------------

        # ini bagian yang diubah
        score = x.reshape(B, N)

        return score