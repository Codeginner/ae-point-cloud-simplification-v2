"""
proposed_method — Point Cloud Simplification via NC-Score Guided DGCNN.

Public API:
    PointCloudSimplifier  : main model (model.py)
    DGCNNEncoder          : encoder backbone (encoder.py)
    EdgeConvLayer         : single EdgeConv block (encoder.py)
    NCScoreModule         : non-local context score (nc_score.py)
    ImportanceScoringMLP  : feature fusion + scoring (scoring.py)
    AdaptiveSelector      : importance-guided point selection (selector.py)
    FoldingNetDecoder     : folding-based reconstruction (decoder.py)
    GeometryAwareLoss     : composite geometry loss (loss.py)
    ChamferLoss           : Chamfer distance component (loss.py)
    NormalConsistencyLoss : normal vector consistency (loss.py)
    NCScorePreservLoss    : NC score preservation (loss.py)
"""

from .model    import PointCloudSimplifier
from .encoder  import DGCNNEncoder, EdgeConvLayer
from .nc_score import NCScoreModule
from .scoring  import ImportanceScoringMLP
from .selector import AdaptiveSelector
from .decoder  import FoldingNetDecoder
#from .visualize import visualize_point_clouds
from .loss     import (
    GeometryAwareLoss,
    ChamferLoss,
    NormalConsistencyLoss,
    NCScorePreservLoss,
)

__all__ = [
    "PointCloudSimplifier",
    "DGCNNEncoder",
    "EdgeConvLayer",
    "NCScoreModule",
    "ImportanceScoringMLP",
    "AdaptiveSelector",
    "FoldingNetDecoder",
    "GeometryAwareLoss",
    "ChamferLoss",
    "NormalConsistencyLoss",
    "NCScorePreservLoss",
    "visualize_point_clouds",
]
