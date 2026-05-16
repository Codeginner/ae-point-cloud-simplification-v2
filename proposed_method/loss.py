"""
loss.py — Geometry-Aware Loss functions.

Components:
    ChamferLoss             : symmetric Chamfer Distance (L_cd)
    NormalConsistencyLoss   : normal vector consistency  (L_n)
    NCScorePreservLoss      : NC score preservation      (L_nc)
    GeometryAwareLoss       : weighted combination of the three above

Total loss:
    L = λ1 * L_cd + λ2 * L_n + λ3 * L_nc
"""

import torch
import torch.nn as nn
from torch import Tensor

from .nc_score import NCScoreModule
from .utils    import knn_graph, index_points


# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

class ChamferLoss(nn.Module):
    """Symmetric Chamfer Distance between two point sets P and Q.

    L_cd = (1/|P|) Σ_{p∈P} min_{q∈Q} ‖p−q‖²
         + (1/|Q|) Σ_{q∈Q} min_{p∈P} ‖q−p‖²

    Args:
        reduction: 'mean' (default) or 'sum' over batch.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def _pairwise_dist(self, A: Tensor, B: Tensor) -> Tensor:
        """Compute pairwise squared Euclidean distances.

        Args:
            A: (B, N, 3)
            B: (B, M, 3)

        Returns:
            dist: (B, N, M)
        """
        inner = -2.0 * torch.bmm(A, B.transpose(2, 1))     # (B, N, M)
        aa    = (A ** 2).sum(dim=-1, keepdim=True)          # (B, N, 1)
        bb    = (B ** 2).sum(dim=-1, keepdim=True)          # (B, M, 1)
        return aa + inner + bb.transpose(2, 1)              # (B, N, M)

    def forward(self, P: Tensor, Q: Tensor) -> Tensor:
        """
        Args:
            P: Reconstructed point cloud  (B, N, 3).
            Q: Ground-truth point cloud   (B, M, 3).

        Returns:
            L_cd: Scalar Chamfer distance.
        """
        dist  = self._pairwise_dist(P, Q)            # (B, N, M)

        p_to_q = dist.min(dim=2).values.mean(dim=1)  # (B,)
        q_to_p = dist.min(dim=1).values.mean(dim=1)  # (B,)

        loss = p_to_q + q_to_p                       # (B,)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ---------------------------------------------------------------------------
# Normal Consistency Loss
# ---------------------------------------------------------------------------

class NormalConsistencyLoss(nn.Module):
    """Normal consistency loss between reconstructed and input point clouds.

    Estimates local normals via PCA on KNN neighbourhoods, then penalises
    angular deviation:

        L_n = (1/M) Σ_i (1 − |n_i · n*_i|)

    Args:
        k:   KNN for normal estimation. Default 10.
        eps: Numerical stability. Default 1e-8.
    """

    def __init__(self, k: int = 10, eps: float = 1e-8) -> None:
        super().__init__()
        self.k   = k
        self.eps = eps

    def _estimate_normals(self, P: Tensor) -> Tensor:
        """Estimate per-point normals via local PCA.

        Args:
            P: (B, N, 3)

        Returns:
            normals: (B, N, 3)  unit vectors.
        """
        B, N, _ = P.shape
        idx         = knn_graph(P, self.k)          # (B, N, k)
        neighbours  = index_points(P, idx)           # (B, N, k, 3)

        # Center neighbours
        centroid    = neighbours.mean(dim=2, keepdim=True)      # (B, N, 1, 3)
        centred     = neighbours - centroid                      # (B, N, k, 3)

        # SVD on centred neighbourhood matrix — lebih stabil di DataParallel
        # centred: (B, N, k, 3) → reshape ke (B*N, k, 3) biar SVD bisa batch
        BN = B * N
        centred_flat = centred.reshape(BN, self.k, 3)           # (B*N, k, 3)

        # torch.linalg.svd: U(B*N,k,k), S(B*N,3), Vh(B*N,3,3)
        # Normal = right singular vector terkecil = baris terakhir Vh
        _, _, Vh = torch.linalg.svd(centred_flat, full_matrices=False)
        normals  = Vh[:, -1, :]                                 # (B*N, 3)
        normals  = normals.reshape(B, N, 3)                     # (B, N, 3)

        # Normalise
        normals = normals / (normals.norm(dim=-1, keepdim=True) + self.eps)
        return normals

    def forward(self, P_recon: Tensor, P_input: Tensor) -> Tensor:
        """
        Args:
            P_recon: Reconstructed cloud  (B, M, 3).
            P_input: Original input cloud (B, N, 3).

        Returns:
            L_n: Scalar normal consistency loss.
        """
        n_recon = self._estimate_normals(P_recon)    # (B, M, 3)
        n_input = self._estimate_normals(P_input)    # (B, N, 3)

        # BUG FIX: use nearest-neighbour matching instead of first-M-points truncation.
        # For each reconstructed point find the spatially closest input point's normal.
        with torch.no_grad():
            diff   = P_recon.unsqueeze(2) - P_input.unsqueeze(1)   # (B, M, N, 3)
            dists  = (diff ** 2).sum(dim=-1)                        # (B, M, N)
            nn_idx = dists.argmin(dim=-1)                           # (B, M)
            nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, 3)     # (B, M, 3)

        n_matched = n_input.gather(1, nn_idx_exp)                   # (B, M, 3)
        dot = (n_recon * n_matched).sum(dim=-1)                     # (B, M)
        L_n = (1.0 - dot.abs()).mean()
        return L_n


# ---------------------------------------------------------------------------
# NC Score Preservation Loss
# ---------------------------------------------------------------------------

class NCScorePreservLoss(nn.Module):
    """NC Score preservation loss.

    Penalises mismatch between NC scores of reconstructed and input points:

        L_nc = (1/M) Σ_i (s(p_i) − s(q*_i))²

    where q*_i is the nearest input point to reconstructed point p_i.

    Args:
        k:   KNN for NC score module. Default 20.
        eps: Numerical stability. Default 1e-8.
    """

    def __init__(self, k: int = 20, eps: float = 1e-8) -> None:
        super().__init__()
        self.nc_scorer = NCScoreModule(k=k, eps=eps)

    def forward(self, P_recon: Tensor, P_input: Tensor) -> Tensor:
        """
        Args:
            P_recon: Reconstructed cloud  (B, M, 3).
            P_input: Original input cloud (B, N, 3).

        Returns:
            L_nc: Scalar NC score preservation loss.
        """
        s_recon = self.nc_scorer(P_recon)   # (B, M)
        s_input = self.nc_scorer(P_input)   # (B, N)

        # BUG FIX: use nearest-neighbour matching instead of first-M-points truncation.
        # For each reconstructed point find its closest input point and use that NC score.
        # P_recon: (B, M, 3),  P_input: (B, N, 3)
        with torch.no_grad():
            diff   = P_recon.unsqueeze(2) - P_input.unsqueeze(1)   # (B, M, N, 3)
            dists  = (diff ** 2).sum(dim=-1)                        # (B, M, N)
            nn_idx = dists.argmin(dim=-1)                           # (B, M)

        # Gather matching NC scores from input cloud
        s_target = s_input.gather(1, nn_idx)                        # (B, M)

        L_nc = ((s_recon - s_target) ** 2).mean()
        return L_nc


# ---------------------------------------------------------------------------
# GeometryAwareLoss  (orchestrator)
# ---------------------------------------------------------------------------

class GeometryAwareLoss(nn.Module):
    """Weighted combination of geometry-aware loss components.

        L_total = λ1·L_cd + λ2·L_n + λ3·L_nc + λ4·L_score

    Components
    ----------
    L_cd    : Chamfer Distance between P_recon and P_input.
    L_n     : Normal Consistency between P_recon and P_input.
    L_nc    : NC Score Preservation (simplified ↔ input NC scores).
    L_score : Score Supervision — auxiliary signal so scorer receives
              gradient even without STE (safety net / warmup helper).

    Note: The primary gradient path to the scorer is now through STE
    in AdaptiveSelector (P_s carries grad through scale factor).
    L_score provides an additional direct signal.

    Args:
        lambda_1: Weight for Chamfer Distance.       Default 1.0.
        lambda_2: Weight for Normal Consistency.     Default 0.5.
        lambda_3: Weight for NC Score Preservation.  Default 0.3.
        lambda_4: Weight for Score Supervision.      Default 0.3.
        k_normal: KNN for normal estimation.         Default 10.
        k_nc    : KNN for NC score computation.      Default 20.
    """

    def __init__(
        self,
        lambda_1: float = 1.0,
        lambda_2: float = 0.5,
        lambda_3: float = 0.3,
        lambda_4: float = 0.3,
        k_normal: int   = 10,
        k_nc:     int   = 20,
    ) -> None:
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

        self.chamfer_loss = ChamferLoss()
        self.normal_loss  = NormalConsistencyLoss(k=k_normal)
        self.nc_loss      = NCScorePreservLoss(k=k_nc)

    def forward(
        self,
        P_recon: Tensor,
        P_input: Tensor,
        P_simplified: Tensor,
        score: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            P_recon      : Reconstructed cloud  (B, M, 3).
            P_input      : Original input cloud (B, N, 3).
            P_simplified : Simplified cloud     (B, M, 3).
            score        : Importance scores    (B, N).

        Returns:
            loss_dict: total, chamfer, normal, nc, score keys.
        """
        L_cd = self.chamfer_loss(P_recon, P_input)
        L_n  = self.normal_loss(P_recon, P_input)
        L_nc = self.nc_loss(P_recon, P_input)

        # ------------------------------------------------------------------
        # L_score : Score Supervision Loss
        #
        # Target: how far is each input point from the nearest simplified
        # point, normalised to [0, 1] per sample.
        # High target → point was far from simplified set → should have been
        # selected → scorer should have given it a high importance score.
        # ------------------------------------------------------------------
        with torch.no_grad():
            diff   = P_input.unsqueeze(2) - P_simplified.unsqueeze(1)  # (B,N,M,3)
            dist   = (diff ** 2).sum(-1).min(dim=2).values              # (B, N)
            d_max  = dist.max(dim=1, keepdim=True).values.clamp(1e-8)
            target = dist / d_max                                        # (B, N) ∈ [0,1]

        L_score = torch.nn.functional.mse_loss(score, target)

        L_total = (
            self.lambda_1 * L_cd
            + self.lambda_2 * L_n
            + self.lambda_3 * L_nc
            + self.lambda_4 * L_score
        )

        return {
            "total":   L_total,
            "chamfer": L_cd,
            "normal":  L_n,
            "nc":      L_nc,
            "score":   L_score,
        }