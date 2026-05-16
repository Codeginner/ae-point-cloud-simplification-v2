"""
evaluate_downstream.py
======================
Downstream classification evaluation for point cloud simplification methods.

Compares:
    1. Ours          — trained PointCloudSimplifier
    2. FPS           — Farthest Point Sampling
    3. Random        — Random Sampling
    4. Voxel         — Voxel Grid Downsampling
    5. APES          — Attention-based Point cloud Edge Sampling
    6. SampleNet     — Learning-based task-driven sampling

Protocol:
    • Pre-trained DGCNN classifier (trained on full 1024-pt clouds)
    • All methods evaluated WITHOUT retraining the classifier
    • Sweeps M ∈ {64, 128, 256, 512, 768}
    • Reports OA (Overall Accuracy) and per-class accuracy

Usage:
    python evaluate_downstream.py \
        --data_root   ./data \
        --simplifier  ./checkpoints/best.pth \
        --classifier  ./checkpoints/dgcnn_cls.pth \
        --M           64 128 256 512 768 \
        --batch_size  32 \
        --out_dir     ./eval_results
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposed_method.train  import PointCloudDataset
from proposed_method.model  import PointCloudSimplifier
from proposed_method.utils  import index_points

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# DGCNN Classifier  (same architecture used for training)
# ===========================================================================

def knn_cls(x: Tensor, k: int) -> Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx    = (x ** 2).sum(1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(2, 1)
    return pairwise.topk(k, dim=-1)[1]


def get_graph_feature(x: Tensor, k: int = 20, idx=None) -> Tensor:
    B, C, N = x.shape
    if idx is None:
        idx = knn_cls(x, k)
    device = x.device
    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x   = x.transpose(2, 1).contiguous()
    feature = x.view(B * N, -1)[idx].view(B, N, k, C)
    x       = x.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat([feature - x, x], dim=3).permute(0, 3, 1, 2)


class DGCNNClassifier(nn.Module):
    """DGCNN classification head — Wang et al. 2019."""
    def __init__(self, num_class: int = 10, k: int = 20, dropout: float = 0.5):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6,   64,  1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2,64,  1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2,128, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2,256,1, bias=False), self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024,1, bias=False), self.bn5, nn.LeakyReLU(0.2))

        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6     = nn.BatchNorm1d(512)
        self.dp1     = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7     = nn.BatchNorm1d(256)
        self.dp2     = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, num_class)

    def forward(self, P: Tensor) -> Tensor:
        """Args: P (B, N, 3) → logits (B, num_class)"""
        x = P.permute(0, 2, 1)           # (B, 3, N)
        B, _, N = x.shape
        x1 = self.conv1(get_graph_feature(x, self.k)).max(-1)[0]
        x2 = self.conv2(get_graph_feature(x1, self.k)).max(-1)[0]
        x3 = self.conv3(get_graph_feature(x2, self.k)).max(-1)[0]
        x4 = self.conv4(get_graph_feature(x3, self.k)).max(-1)[0]
        x  = self.conv5(torch.cat([x1, x2, x3, x4], 1))
        x  = torch.cat([x.max(-1)[0], x.mean(-1)], 1)
        x  = self.dp1(F.leaky_relu(self.bn6(self.linear1(x)), 0.2))
        x  = self.dp2(F.leaky_relu(self.bn7(self.linear2(x)), 0.2))
        return self.linear3(x)


# ===========================================================================
# Baseline Sampling Methods
# ===========================================================================

def random_sample(P: Tensor, M: int) -> Tensor:
    """Random sampling. P: (B,N,3) → (B,M,3)"""
    B, N, _ = P.shape
    idx = torch.stack([torch.randperm(N, device=P.device)[:M] for _ in range(B)])
    return P.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps(P: Tensor, M: int) -> Tensor:
    """Farthest Point Sampling. P: (B,N,3) → (B,M,3)"""
    B, N, _ = P.shape
    device   = P.device
    selected = torch.zeros(B, M, dtype=torch.long, device=device)
    dist     = torch.full((B, N), float("inf"), device=device)
    # Start from a random point per sample
    selected[:, 0] = torch.randint(N, (B,), device=device)
    for i in range(1, M):
        last = P.gather(1, selected[:, i-1:i].unsqueeze(-1).expand(-1, -1, 3))
        d    = ((P - last) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        selected[:, i] = dist.argmax(dim=1)
    return P.gather(1, selected.unsqueeze(-1).expand(-1, -1, 3))


def voxel_downsample(P: Tensor, M: int) -> Tensor:
    """
    Voxel grid downsampling — approximated to output exactly M points.
    Uses adaptive voxel size: binary search until we get >= M points,
    then randomly pick M from the voxel centres.
    P: (B,N,3) → (B,M,3)
    """
    B, N, C = P.shape
    results = []
    for b in range(B):
        pts = P[b]  # (N, 3)
        mn  = pts.min(0).values
        mx  = pts.max(0).values
        span = (mx - mn).max().item()

        # Binary search for voxel size
        lo, hi = span / N, span
        for _ in range(20):
            mid    = (lo + hi) / 2
            voxels = ((pts - mn) / mid).long()
            unique = torch.unique(voxels, dim=0)
            if unique.shape[0] >= M:
                hi = mid
            else:
                lo = mid
            if abs(unique.shape[0] - M) <= 2:
                break

        # Compute voxel centroids
        voxel_ids = ((pts - mn) / hi).long()
        key = voxel_ids[:, 0] * 100003 + voxel_ids[:, 1] * 1003 + voxel_ids[:, 2]
        _, inv = torch.unique(key, return_inverse=True)
        n_vox  = inv.max().item() + 1
        centres = torch.zeros(n_vox, 3, device=pts.device)
        counts  = torch.zeros(n_vox, 1, device=pts.device)
        centres.scatter_add_(0, inv.unsqueeze(-1).expand(-1, 3), pts)
        counts.scatter_add_(0, inv.unsqueeze(-1), torch.ones(N, 1, device=pts.device))
        centres = centres / counts.clamp(1)

        # Sample M from voxel centres
        if centres.shape[0] >= M:
            idx = torch.randperm(centres.shape[0], device=pts.device)[:M]
            results.append(centres[idx])
        else:
            # Pad by repeating
            rep = M - centres.shape[0]
            pad = centres[torch.randint(centres.shape[0], (rep,), device=pts.device)]
            results.append(torch.cat([centres, pad], 0))

    return torch.stack(results)  # (B, M, 3)


# ===========================================================================
# APES — Attention-based Point cloud Edge Sampling
# Kim et al. 2023 — simplified re-implementation
# Paper: https://arxiv.org/abs/2302.14673
# ===========================================================================

class APES(nn.Module):
    """
    Attention-based Point cloud Edge Sampling.

    Architecture (simplified faithful re-impl):
        1. Local geometry descriptor via KNN + MLP          → per-point feat
        2. Edge attention score from feat difference        → per-point scalar
        3. Top-M selection by attention score
    """
    def __init__(self, k: int = 20, feat_dim: int = 64):
        super().__init__()
        self.k = k
        # Local geometry MLP: 6 (concat of xi and xi-xj) → feat_dim
        self.local_mlp = nn.Sequential(
            nn.Conv2d(6, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        # Attention scorer: feat_dim → 1
        self.attn = nn.Sequential(
            nn.Conv1d(feat_dim, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, 1),
        )

    def _edge_feature(self, P: Tensor) -> Tensor:
        """Build edge features from KNN. P:(B,N,3) → (B,6,N,k)"""
        x   = P.permute(0, 2, 1)          # (B,3,N)
        idx = knn_cls(x, self.k)
        B, C, N = x.shape
        device  = x.device
        base    = torch.arange(B, device=device).view(-1,1,1) * N
        flat    = (idx + base).view(-1)
        x_t     = x.permute(0,2,1).contiguous().view(B*N, C)
        neigh   = x_t[flat].view(B, N, self.k, C).permute(0,3,1,2)  # (B,C,N,k)
        xi      = x.unsqueeze(-1).expand_as(neigh)
        return torch.cat([xi, neigh - xi], dim=1)                    # (B,6,N,k)

    def get_scores(self, P: Tensor) -> Tensor:
        """Compute per-point attention scores. P:(B,N,3) → (B,N)"""
        edge  = self._edge_feature(P)                 # (B,6,N,k)
        feat  = self.local_mlp(edge).max(-1)[0]       # (B,feat_dim,N)
        score = self.attn(feat).squeeze(1)            # (B,N)
        return score

    def forward(self, P: Tensor, M: int) -> Tensor:
        """Select M points. P:(B,N,3) → (B,M,3)"""
        score = self.get_scores(P)
        idx   = score.topk(M, dim=1)[1]               # (B,M)
        return P.gather(1, idx.unsqueeze(-1).expand(-1,-1,3))


# ===========================================================================
# SampleNet — Task-driven soft sampling
# Lang et al. 2020 — simplified re-implementation
# Paper: https://arxiv.org/abs/1912.03663
# ===========================================================================

class SampleNet(nn.Module):
    """
    SampleNet: Differentiable Point Cloud Sampling.

    Simplified faithful re-impl:
        1. PointNet encoder → global feature z
        2. FC decoder → M projected points (soft samples)
        3. Inference: snap projected points to nearest real input point
    """
    def __init__(self, in_dim: int = 3, feat_dim: int = 128):
        super().__init__()
        # Per-point encoding
        self.enc = nn.Sequential(
            nn.Conv1d(in_dim, 64,  1, bias=False), nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Conv1d(64,     128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, feat_dim, 1, bias=False), nn.BatchNorm1d(feat_dim), nn.ReLU(True),
        )
        # This will be built lazily per M value to handle variable M
        self._M     = None
        self._dec   = None
        self.feat_dim = feat_dim

    def _build_decoder(self, M: int, device):
        self._M   = M
        self._dec = nn.Sequential(
            nn.Linear(self.feat_dim, 256), nn.ReLU(True),
            nn.Linear(256, 256),           nn.ReLU(True),
            nn.Linear(256, M * 3),
        ).to(device)

    def forward(self, P: Tensor, M: int) -> Tensor:
        """P:(B,N,3) → (B,M,3) — nearest real point to each projected sample"""
        B, N, _ = P.shape
        if self._M != M:
            self._build_decoder(M, P.device)

        x   = P.permute(0,2,1)                           # (B,3,N)
        f   = self.enc(x).max(-1)[0]                     # (B,feat_dim)
        proj = self._dec(f).view(B, M, 3)                # (B,M,3)

        # Snap to nearest real point (inference)
        with torch.no_grad():
            diff = proj.unsqueeze(2) - P.unsqueeze(1)    # (B,M,N,3)
            idx  = (diff**2).sum(-1).argmin(-1)          # (B,M)
        return P.gather(1, idx.unsqueeze(-1).expand(-1,-1,3))


# ===========================================================================
# Evaluator
# ===========================================================================

@torch.no_grad()
def evaluate_one(
    classifier: nn.Module,
    loader: DataLoader,
    simplify_fn,
    M: int,
    device: torch.device,
    num_class: int = 10,
) -> dict:
    """
    Run one evaluation pass.

    Args:
        simplify_fn: callable(P: Tensor, M: int) → P_simplified: Tensor
                     P is (B, N, 3), output must be (B, M, 3).

    Returns:
        dict with overall_acc, per_class_acc, avg_time_ms
    """
    classifier.eval()
    all_preds, all_labels = [], []
    times = []

    for P, labels in loader:
        P, labels = P.to(device), labels.to(device)
        t0   = time.perf_counter()
        P_s  = simplify_fn(P, M)
        times.append((time.perf_counter() - t0) / P.shape[0] * 1000)
        logits = classifier(P_s)
        preds  = logits.argmax(-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    oa     = (preds == labels).float().mean().item()

    per_class = []
    for c in range(num_class):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append((preds[mask] == labels[mask]).float().mean().item())

    return {
        "overall_acc":   round(oa * 100, 2),
        "per_class_acc": round(np.mean(per_class) * 100, 2),
        "avg_time_ms":   round(np.mean(times), 3),
    }


# ===========================================================================
# Training helper for APES & SampleNet
# ===========================================================================

def train_sampler(
    model: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    M: int,
    device: torch.device,
    epochs: int = 20,
):
    """
    Quick task-driven training for APES/SampleNet.
    Frozen classifier, only sampler trained.
    """
    model.train()
    for p in classifier.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4
    )
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total, correct, n = 0.0, 0, 0
        for P, labels in train_loader:
            P, labels = P.to(device), labels.to(device)
            opt.zero_grad()
            if isinstance(model, SampleNet):
                P_s = model(P, M)
            else:
                P_s = model(P, M)
            logits = classifier(P_s)
            loss   = ce(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total   += loss.item() * P.shape[0]
            correct += (logits.argmax(-1) == labels).sum().item()
            n       += P.shape[0]
        if (ep + 1) % 5 == 0:
            logger.info(f"  sampler training ep {ep+1}/{epochs}  "
                        f"loss={total/n:.4f}  acc={correct/n*100:.1f}%")

    for p in classifier.parameters():
        p.requires_grad_(True)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="./data")
    p.add_argument("--simplifier",  default="./checkpoints/best.pth",
                   help="Path ke best.pth dari method kamu")
    p.add_argument("--classifier",  default=None,
                   help="Path ke pre-trained DGCNN classifier. "
                        "Jika None, classifier ditraining dari scratch "
                        "pada cloud penuh (butuh --cls_epochs).")
    p.add_argument("--M",           nargs="+", type=int,
                   default=[64, 128, 256, 512, 768])
    p.add_argument("--n_points",    type=int, default=1024)
    p.add_argument("--num_class",   type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cls_epochs",  type=int, default=100,
                   help="Epoch training classifier jika --classifier tidak disediakan")
    p.add_argument("--sampler_epochs", type=int, default=30,
                   help="Epoch training APES/SampleNet")
    p.add_argument("--simplifier_M",   type=int, default=512,
                   help="M yang dipakai saat training simplifier (untuk load checkpoint)")
    p.add_argument("--out_dir",     default="./eval_results")
    return p.parse_args()


def train_classifier(classifier, train_loader, device, epochs=100):
    logger.info(f"Training DGCNN classifier for {epochs} epochs on full clouds...")
    opt  = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce   = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        classifier.train()
        for P, labels in train_loader:
            P, labels = P.to(device), labels.to(device)
            opt.zero_grad()
            ce(classifier(P), labels).backward()
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            logger.info(f"  classifier ep {ep+1}/{epochs}")
    return classifier


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds = PointCloudDataset(args.data_root, "train", args.n_points, augment=True)
    test_ds  = PointCloudDataset(args.data_root, "test",  args.n_points, augment=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    # ── Classifier ─────────────────────────────────────────────────────────
    classifier = DGCNNClassifier(num_class=args.num_class).to(device)
    if args.classifier and Path(args.classifier).exists():
        ckpt = torch.load(args.classifier, map_location=device)
        state = ckpt.get("model", ckpt)
        classifier.load_state_dict(state)
        logger.info(f"Loaded classifier from {args.classifier}")
    else:
        logger.info("No classifier checkpoint — training from scratch on full clouds")
        train_classifier(classifier, train_loader, device, args.cls_epochs)
        torch.save({"model": classifier.state_dict()},
                   out_dir / "dgcnn_cls.pth")
        logger.info(f"Saved trained classifier → {out_dir}/dgcnn_cls.pth")
    classifier.eval()

    # ── Our method ─────────────────────────────────────────────────────────
    our_model = PointCloudSimplifier(M=args.simplifier_M).to(device)
    ckpt      = torch.load(args.simplifier, map_location=device)
    our_model.load_state_dict(ckpt.get("model", ckpt))
    our_model.eval()

    @torch.no_grad()
    def our_simplify(P: Tensor, M: int) -> Tensor:
        """Use our model to select M points (re-select with topk if M != trained M)"""
        out   = our_model(P, compute_loss=False)
        P_s   = out["P_simplified"]       # (B, trained_M, 3)
        # If requested M differs from trained M, subsample/FPS from simplified
        if P_s.shape[1] == M:
            return P_s
        elif P_s.shape[1] > M:
            return fps(P_s, M)
        else:
            # Shouldn't happen if M <= simplifier_M; fallback to FPS on original
            return fps(P, M)

    # ── Learning-based baselines (need training) ────────────────────────────
    apes_model   = APES().to(device)
    samplenet    = SampleNet().to(device)

    logger.info("Training APES ...")
    # APES trained per M value for fairness
    apes_models = {}
    for M_val in args.M:
        m = APES().to(device)
        train_sampler(m, classifier, train_loader, M_val, device, args.sampler_epochs)
        m.eval()
        apes_models[M_val] = m

    logger.info("Training SampleNet ...")
    samplenet_models = {}
    for M_val in args.M:
        m = SampleNet().to(device)
        train_sampler(m, classifier, train_loader, M_val, device, args.sampler_epochs)
        m.eval()
        samplenet_models[M_val] = m

    # ── Evaluation sweep ───────────────────────────────────────────────────
    methods = {
        "ours":      our_simplify,
        "fps":       fps,
        "random":    random_sample,
        "voxel":     voxel_downsample,
        "apes":      None,    # will use apes_models[M] per M
        "samplenet": None,
    }

    results = {m: {} for m in methods}

    for M_val in args.M:
        logger.info(f"\n{'='*50}")
        logger.info(f"  M = {M_val}")
        logger.info(f"{'='*50}")

        methods["apes"]      = lambda P, M, _m=M_val: apes_models[_m](P, M)
        methods["samplenet"] = lambda P, M, _m=M_val: samplenet_models[_m](P, M)

        for name, fn in methods.items():
            logger.info(f"  Evaluating {name} ...")
            r = evaluate_one(classifier, test_loader, fn, M_val, device, args.num_class)
            results[name][M_val] = r
            logger.info(f"  {name:12s}  OA={r['overall_acc']:.2f}%  "
                        f"mAcc={r['per_class_acc']:.2f}%  "
                        f"time={r['avg_time_ms']:.1f}ms")

    # ── Save results ───────────────────────────────────────────────────────
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"M_values": args.M, "results": results}, f, indent=2)
    logger.info(f"\nResults saved → {out_json}")

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Method':<12}", end="")
    for M_val in args.M:
        print(f"  M={M_val:<6}", end="")
    print()
    print("-"*70)
    for name in methods:
        print(f"{name:<12}", end="")
        for M_val in args.M:
            r = results[name].get(M_val, {})
            print(f"  {r.get('overall_acc', 0):>5.1f}%  ", end="")
        print()
    print("="*70)

    logger.info(f"\nDone. All results in {out_dir}/")


if __name__ == "__main__":
    main()
