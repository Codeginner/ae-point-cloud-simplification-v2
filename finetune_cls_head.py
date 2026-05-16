"""
finetune_cls_head.py
====================
Train a lightweight classification head on top of frozen simplified points.
Simplifier weights are never updated.

Pipeline:
    P_input (1024 pts)
        → Simplifier [FROZEN] → P_s (M pts)
        → ClsHead [trained]   → class logits

Usage:
    python finetune_cls_head.py \
        --simplifier ./checkpoints/best.pth \
        --data_root  ./data \
        --M          64 128 256 512 768 \
        --epochs     100 \
        --out_dir    ./cls_head_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposed_method.train import PointCloudDataset
from proposed_method.model import PointCloudSimplifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Lightweight Classification Head
# DGCNN-lite: 3 EdgeConv layers + global pool + 2-layer MLP
# Deliberately small so it learns FROM the simplified points,
# not in spite of them.
# ===========================================================================

def knn(x: Tensor, k: int) -> Tensor:
    """x: (B,3,N) → idx: (B,N,k)"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx    = (x ** 2).sum(1, keepdim=True)
    dist  = -xx - inner - xx.transpose(2, 1)
    return dist.topk(k, dim=-1)[1]


def edge_feature(x: Tensor, k: int) -> Tensor:
    """Build edge features. x:(B,C,N) → (B,2C,N,k)"""
    B, C, N = x.shape
    idx     = knn(x[:, :3], k)                              # use xyz for knn
    base    = torch.arange(B, device=x.device).view(-1,1,1) * N
    flat    = (idx + base).view(-1)
    x_t     = x.permute(0,2,1).contiguous().view(B*N, C)
    neigh   = x_t[flat].view(B, N, k, C).permute(0,3,1,2)  # (B,C,N,k)
    xi      = x.unsqueeze(-1).expand_as(neigh)
    return torch.cat([xi, neigh - xi], 1)                   # (B,2C,N,k)


class ClsHead(nn.Module):
    """
    Lightweight DGCNN-style head.
    Input : P_s  (B, M, 3)   — simplified points (frozen, from simplifier)
    Output: logits (B, num_class)
    """
    def __init__(self, num_class: int = 10, k: int = 16, dropout: float = 0.4):
        super().__init__()
        self.k = k
        # EdgeConv blocks
        self.ec1 = nn.Sequential(
            nn.Conv2d(6,   64, 1, bias=False), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2))
        self.ec2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2))
        self.ec3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # Aggregation
        self.agg = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        # Classifier MLP
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(256, num_class))

    def forward(self, P_s: Tensor) -> Tensor:
        """P_s: (B, M, 3) → logits: (B, num_class)"""
        x = P_s.permute(0, 2, 1)                 # (B, 3, M)

        x1 = self.ec1(edge_feature(x,  self.k)).max(-1)[0]   # (B, 64, M)
        x2 = self.ec2(edge_feature(x1, self.k)).max(-1)[0]   # (B, 64, M)
        x3 = self.ec3(
            edge_feature(torch.cat([x1, x2], 1), self.k)
        ).max(-1)[0]                                           # (B,128, M)

        feat = self.agg(torch.cat([x1, x2, x3], 1))           # (B,512, M)
        g    = torch.cat([feat.max(-1)[0], feat.mean(-1)], 1) # (B,1024)
        return self.mlp(g)                                     # (B, num_class)


# ===========================================================================
# Training & evaluation
# ===========================================================================

def train_head(
    simplifier: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    M: int,
    device: torch.device,
) -> tuple[float, float]:
    head.train()
    total_loss, correct, n = 0.0, 0, 0
    ce = nn.CrossEntropyLoss()

    for P, labels in loader:
        P, labels = P.to(device), labels.to(device)

        # Simplified points — no grad through simplifier
        with torch.no_grad():
            out = simplifier(P, compute_loss=False)
            P_s = out["P_simplified"]        # (B, trained_M, 3)

            # If head M != simplifier M, subsample via FPS
            if P_s.shape[1] != M:
                P_s = fps_subsample(P_s, M)

        optimizer.zero_grad()
        logits = head(P_s)
        loss   = ce(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * P.shape[0]
        correct    += (logits.argmax(-1) == labels).sum().item()
        n          += P.shape[0]

    scheduler.step()
    return total_loss / n, correct / n * 100


@torch.no_grad()
def eval_head(
    simplifier: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    M: int,
    device: torch.device,
    num_class: int = 10,
) -> dict:
    head.eval()
    all_preds, all_labels = [], []

    for P, labels in loader:
        P = P.to(device)
        out = simplifier(P, compute_loss=False)
        P_s = out["P_simplified"]
        if P_s.shape[1] != M:
            P_s = fps_subsample(P_s, M)
        preds = head(P_s).argmax(-1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    oa     = (preds == labels).float().mean().item() * 100

    per_class = []
    for c in range(num_class):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append((preds[mask] == c).float().mean().item())

    return {
        "overall_acc":   round(oa, 2),
        "per_class_acc": round(sum(per_class) / len(per_class) * 100, 2),
    }


def fps_subsample(P: Tensor, M: int) -> Tensor:
    """FPS from trained_M → M. P:(B,N,3) → (B,M,3)"""
    B, N, _ = P.shape
    if M >= N:
        return P
    device   = P.device
    selected = torch.zeros(B, M, dtype=torch.long, device=device)
    dist     = torch.full((B, N), float("inf"), device=device)
    selected[:, 0] = torch.randint(N, (B,), device=device)
    for i in range(1, M):
        last = P.gather(1, selected[:, i-1:i].unsqueeze(-1).expand(-1,-1,3))
        d    = ((P - last) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        selected[:, i] = dist.argmax(1)
    return P.gather(1, selected.unsqueeze(-1).expand(-1,-1,3))


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--simplifier",    default="./checkpoints/best.pth")
    p.add_argument("--data_root",     default="./data")
    p.add_argument("--M",             nargs="+", type=int,
                   default=[64, 128, 256, 512, 768])
    p.add_argument("--simplifier_M",  type=int, default=512,
                   help="M used when training the simplifier")
    p.add_argument("--n_points",      type=int, default=1024)
    p.add_argument("--num_class",     type=int, default=10)
    p.add_argument("--epochs",        type=int, default=100)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--out_dir",       default="./cls_head_results")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds = PointCloudDataset(args.data_root, "train", args.n_points, augment=True)
    test_ds  = PointCloudDataset(args.data_root, "test",  args.n_points, augment=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    # ── Simplifier (frozen) ────────────────────────────────────────────────
    simplifier = PointCloudSimplifier(M=args.simplifier_M).to(device)
    ckpt       = torch.load(args.simplifier, map_location=device)
    simplifier.load_state_dict(ckpt.get("model", ckpt))
    simplifier.eval()
    for p in simplifier.parameters():
        p.requires_grad_(False)
    logger.info(f"Simplifier loaded + frozen: {args.simplifier}")

    # ── Train one head per M ───────────────────────────────────────────────
    results = {}

    for M_val in args.M:
        logger.info(f"\n{'='*50}")
        logger.info(f"  Training cls head for M={M_val}")
        logger.info(f"{'='*50}")

        k    = min(16, M_val - 1)   # knn k must be < M
        head = ClsHead(num_class=args.num_class, k=k).to(device)

        opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)

        best_acc  = 0.0
        best_state = None

        for ep in range(args.epochs):
            tr_loss, tr_acc = train_head(
                simplifier, head, train_loader, opt, sch, M_val, device)

            if (ep + 1) % 20 == 0:
                val = eval_head(simplifier, head, test_loader,
                                M_val, device, args.num_class)
                logger.info(f"  ep {ep+1:3d}/{args.epochs}  "
                            f"loss={tr_loss:.4f}  train_acc={tr_acc:.1f}%  "
                            f"val_OA={val['overall_acc']:.2f}%")

                if val["overall_acc"] > best_acc:
                    best_acc   = val["overall_acc"]
                    best_state = {k: v.cpu().clone()
                                  for k, v in head.state_dict().items()}

        # Final eval with best weights
        head.load_state_dict(best_state)
        final = eval_head(simplifier, head, test_loader,
                          M_val, device, args.num_class)
        results[M_val] = final

        # Save head checkpoint
        torch.save({"model": head.state_dict(), "M": M_val},
                   out_dir / f"cls_head_M{M_val}.pth")

        logger.info(f"  ★ M={M_val}  OA={final['overall_acc']:.2f}%  "
                    f"mAcc={final['per_class_acc']:.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    out_json = out_dir / "cls_results.json"
    with open(out_json, "w") as f:
        json.dump({"M_values": args.M, "results": results}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"{'M':<8} {'OA':>10} {'mAcc':>10}")
    print(f"{'-'*55}")
    for M_val in args.M:
        r = results[M_val]
        print(f"M={M_val:<5}  {r['overall_acc']:>8.2f}%  {r['per_class_acc']:>8.2f}%")
    print(f"{'='*55}")
    logger.info(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
