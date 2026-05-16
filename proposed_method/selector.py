"""
selector.py — Adaptive Geometry-Balanced Selector with STE.

Implements Eq. (10)–(14) from the proposed method.

Key improvement over the original:
    The original used a Python for-loop over the batch and hard top-k,
    making it:
        1. Slow — serialised over batch, not parallelisable on GPU
        2. Non-differentiable — gradient from reconstruction never
           reached the ImportanceScoringMLP

    This version:
        1. Fully vectorised — no Python loop, runs entirely on GPU
        2. Differentiable via Straight-Through Estimator (STE)
           Forward : hard top-k selection (same discrete output)
           Backward: gradient flows through soft scores to the scorer

    STE trick used here:
        scale = score_selected / score_selected.detach()   ≈ 1.0
        P_s   = P_s_hard * scale.unsqueeze(-1)
        → values unchanged in forward, gradient reaches scorer in backward

Class:
    AdaptiveSelector
"""

import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveSelector(nn.Module):
    """
    Adaptive Geometry-Balanced Selector (vectorised + STE).

    Args:
        M         : Number of output simplified points.
        alpha     : Fraction allocated to contour region. Default 0.7.
        threshold : NC threshold separating contour vs flat. Default 0.5.
    """

    def __init__(
        self,
        M: int = 1024,
        alpha: float = 0.7,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.threshold = threshold

    def forward(
        self,
        P: Tensor,
        score: Tensor,
        nc_score: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Select M points using geometry-balanced selection with STE.

        Args:
            P        : Input point cloud   (B, N, 3)
            score    : Importance scores   (B, N)   — requires_grad=True
            nc_score : NC scores           (B, N)

        Returns:
            idx : Selected indices         (B, M)   — for gathering features
            P_s : Simplified point cloud   (B, M, 3) — differentiable via STE
        """

        B, N, _ = P.shape
        M_c = int(self.alpha * self.M)
        M_f = self.M - M_c

        # ------------------------------------------------------------------
        # 1. Build masked scores — vectorised, no Python loop
        #    Contour mask applied to score: flat points get -inf for contour
        #    pool, contour points get -inf for flat pool.
        # ------------------------------------------------------------------

        contour_mask = nc_score >= self.threshold          # (B, N)

        NEG_INF = torch.finfo(score.dtype).min

        # Scores visible to contour pool (flat points blocked)
        s_contour = score.masked_fill(~contour_mask, NEG_INF)
        # Scores visible to flat pool (contour points blocked)
        s_flat    = score.masked_fill(contour_mask,  NEG_INF)

        # Fallback: if an entire sample has all -inf in one pool, use full score
        all_flat    = (~contour_mask).all(dim=1, keepdim=True)   # (B,1) bool
        all_contour = contour_mask.all(dim=1, keepdim=True)
        s_contour = torch.where(all_flat,    score, s_contour)
        s_flat    = torch.where(all_contour, score, s_flat)

        # ------------------------------------------------------------------
        # 2. Hard top-k selection (non-differentiable, used in forward only)
        # ------------------------------------------------------------------

        _, idx_c = s_contour.topk(M_c, dim=1, largest=True, sorted=False)  # (B, M_c)
        _, idx_f = s_flat.topk(M_f,    dim=1, largest=True, sorted=False)  # (B, M_f)

        idx = torch.cat([idx_c, idx_f], dim=1)   # (B, M)

        # ------------------------------------------------------------------
        # 3. Gather simplified points (hard, non-differentiable path)
        # ------------------------------------------------------------------

        P_s_hard = P.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, M, 3)

        # ------------------------------------------------------------------
        # 4. Straight-Through Estimator (STE)
        #
        #    Problem  : P_s_hard = index_points(P, idx)
        #               The integer-index gather has no gradient w.r.t. score.
        #               Scorer MLP never receives loss signal from reconstruction.
        #
        #    Solution : Multiply P_s by a scale factor that is 1.0 in the
        #               forward pass but carries gradient in the backward pass.
        #
        #       selected_scores = score.gather(1, idx)          (B, M)
        #       scale           = s / s.detach()                ≈ 1.0
        #       P_s             = P_s_hard * scale.unsqueeze(-1)
        #
        #    → Forward : P_s == P_s_hard  (numerically identical)
        #    → Backward: dL/d(score_i) ≠ 0  for all selected i
        #                Gradient flows: loss → P_recon → decoder
        #                                     → P_s → scale → score → scorer
        # ------------------------------------------------------------------

        selected_scores = score.gather(1, idx)                          # (B, M)
        scale = selected_scores / selected_scores.detach().clamp(1e-8)  # (B, M) ≈ 1.0
        P_s = P_s_hard * scale.unsqueeze(-1)                            # (B, M, 3)

        return idx, P_s
