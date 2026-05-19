"""
decoder.py — FoldingNet-based point cloud decoder.

Given:
    • simplified point cloud P_s
    • simplified point features f_s

the decoder:
    1. extracts a global latent vector z
    2. tiles z with a 2D grid
    3. applies two folding stages
    4. reconstructs the point cloud P_recon

Class:
    FoldingNetDecoder
"""

import torch
import torch.nn as nn
from torch import Tensor


class FoldingNetDecoder(nn.Module):
    """
    Two-stage FoldingNet decoder.

    Proposed-method version:

        P_s + f_s
            ↓
        global latent vector z
            ↓
        2D grid folding
            ↓
        reconstructed point cloud

    Pipeline:
        1. MaxPool over simplified features
        2. Build 2D grid
        3. Folding stage 1
        4. Folding stage 2

    Args:
        M:
            Number of reconstruction points.

        in_dim:
            Simplified feature dimension.

            Proposed method:
                448

        latent_dim:
            Global latent vector dimension.

            Proposed method:
                1024
    """

    def __init__(
        self,

        M: int = 1024,

        # ini bagian yang diubah
        in_dim: int = 448,

        latent_dim: int = 1024,
    ) -> None:

        super().__init__()

        self.M = M
        self.latent_dim = latent_dim

        # --------------------------------------------------------------
        # Global feature encoder
        # (B,M,448) -> (B,M,1024)
        # --------------------------------------------------------------

        self.global_mlp = nn.Sequential(

            # ini bagian yang diubah
            nn.Linear(in_dim, latent_dim),

            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------------------
        # BUG FIX: seed_dim changed from 2 (fixed 2D grid) to 3 (P_s xyz).
        # The decoder now uses the actual simplified point positions as fold
        # seeds instead of a canonical 2D grid, so the output is anchored to
        # the correct 3D orientation of the input cloud.
        # --------------------------------------------------------------

        seed_dim = 3   # x, y, z from P_s

        # --------------------------------------------------------------
        # Folding stages
        # --------------------------------------------------------------

        self.fold_layers = nn.ModuleList([

            # ----------------------------------------------------------
            # Fold Stage 1
            # concat(P_s_xyz, z)   — was concat(2D_grid, z)
            # ----------------------------------------------------------

            nn.Sequential(

                nn.Conv1d(
                    seed_dim + latent_dim,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    3,
                    1
                ),
            ),

            # ----------------------------------------------------------
            # Fold Stage 2
            # concat(fold1, z)  — unchanged
            # ----------------------------------------------------------

            nn.Sequential(

                nn.Conv1d(
                    3 + latent_dim,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    512,
                    1
                ),

                nn.ReLU(inplace=True),

                nn.Conv1d(
                    512,
                    3,
                    1
                ),
            ),
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P_s: Tensor,
        f_s: Tensor,
    ) -> Tensor:
        """
        Args:
            P_s:
                Simplified point cloud
                shape = (B,M,3)

            f_s:
                Simplified point features
                shape = (B,M,448)

        Returns:
            P_recon:
                Reconstructed point cloud
                shape = (B,M,3)
        """

        B, M, _ = f_s.shape

        device = f_s.device

        # --------------------------------------------------------------
        # 1. Global latent vector
        # Eq. (15)
        # --------------------------------------------------------------

        # ini bagian yang diubah
        z = self.global_mlp(f_s)

        # MaxPool:
        # (B,M,1024) -> (B,1024)

        # ini bagian yang diubah
        z = z.max(dim=1).values

        # --------------------------------------------------------------
        # 2. Tile latent vector
        # (B,1024) -> (B,1024,M)
        # --------------------------------------------------------------

        # ini bagian yang diubah
        z_tiled = z.unsqueeze(-1).expand(
            -1,
            -1,
            self.M
        )

        # --------------------------------------------------------------
        # BUG FIX: use P_s xyz as the fold seed instead of a fixed 2D
        # grid.  The original code built a canonical grid in [-0.5, 0.5]²
        # which has no orientation information — because DGCNN is
        # approximately rotation-invariant the decoder always reconstructed
        # in a fixed canonical pose, regardless of how the input was
        # oriented.
        #
        # Now:
        #   seed = P_s.permute(0,2,1)   shape: (B, 3, M)
        #
        # Stage 1 input: concat(P_s_xyz, z_tiled)  → (B, 3+1024, M)
        # Stage 2 input: concat(fold1_out, z_tiled) → (B, 3+1024, M)
        # Final output:  fold2_out + P_s_transposed  (residual)
        #   → the network predicts per-point *offsets* from P_s, not
        #     absolute positions.  This makes learning easier and keeps
        #     the reconstruction anchored at the correct 3D location.
        # --------------------------------------------------------------

        seed = P_s.permute(0, 2, 1)        # (B, 3, M)

        # --------------------------------------------------------------
        # 3. Folding Stage 1
        # concat(P_s_xyz, z_tiled) → (B, 3+1024, M)
        # --------------------------------------------------------------

        fold1_input = torch.cat(
            [seed, z_tiled],
            dim=1
        )

        fold1_out = self.fold_layers[0](fold1_input)   # (B, 3, M)

        # --------------------------------------------------------------
        # 4. Folding Stage 2
        # concat(fold1, z_tiled) → (B, 3+1024, M)
        # --------------------------------------------------------------

        fold2_input = torch.cat(
            [fold1_out, z_tiled],
            dim=1
        )

        fold2_out = self.fold_layers[1](fold2_input)   # (B, 3, M)

        # --------------------------------------------------------------
        # 5. Residual: decode offsets from P_s, not absolute positions
        # P_recon = P_s + predicted_offsets
        # --------------------------------------------------------------

        P_recon = fold2_out + seed         # (B, 3, M)

        P_recon = P_recon.permute(0, 2, 1)  # (B, M, 3)

        return P_recon