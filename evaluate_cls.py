"""
evaluate_cls.py
===============
Evaluasi akurasi klasifikasi langsung dari model joint (logits sudah ada di dalam model).
Tidak perlu head terpisah.

Usage:
    python evaluate_cls.py \
        --checkpoint ./checkpoints/best.pth \
        --data_root  ./data \
        --M          512 \
        --num_class  10
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposed_method.train import PointCloudDataset
from proposed_method.model import PointCloudSimplifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, loader, device, num_class, tta=False, tta_runs=10):
    model.eval()
    all_preds, all_labels = [], []

    for P, labels in loader:
        P      = P.to(device)
        labels = labels.to(device)

        if tta:
            # Test-Time Augmentation: average logits over random rotations
            logits_sum = None
            for _ in range(tta_runs):
                # Random rotation around Y axis
                theta = torch.rand(1).item() * 2 * 3.14159
                cos_t, sin_t = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
                rot = torch.tensor([[cos_t, 0, sin_t],
                                    [0,     1, 0    ],
                                    [-sin_t,0, cos_t]], device=device)
                P_aug = P @ rot.T

                # Random jitter
                P_aug = P_aug + torch.randn_like(P_aug) * 0.01

                out = model(P_aug, labels=None, compute_loss=False)
                logits_sum = out["logits"] if logits_sum is None else logits_sum + out["logits"]

            preds = logits_sum.argmax(dim=-1)
        else:
            out   = model(P, labels=None, compute_loss=False)
            preds = out["logits"].argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    oa = (preds == labels).float().mean().item() * 100

    per_class = []
    for c in range(num_class):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append((preds[mask] == c).float().mean().item() * 100)

    return {
        "overall_acc":   round(oa, 2),
        "per_class_acc": round(np.mean(per_class), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./checkpoints/best.pth")
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--M",          type=int, default=512)
    p.add_argument("--num_class",  type=int, default=10)
    p.add_argument("--n_points",   type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--runs",       type=int, default=5,
                   help="Jumlah run untuk averaging (atasi non-determinism)")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    # ── Determinism ────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  seed={args.seed}  runs={args.runs}")

    # Dataset
    test_ds = PointCloudDataset(args.data_root, "test", args.n_points, augment=False)
    loader  = DataLoader(test_ds, args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Test samples: {len(test_ds)}")

    # Model
    model = PointCloudSimplifier(M=args.M, num_class=args.num_class).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    logger.info(f"Loaded: {args.checkpoint}")

    # Multi-run evaluation
    oa_list, macc_list = [], []
    for i in range(args.runs):
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        r = evaluate(model, loader, device, args.num_class)
        oa_list.append(r["overall_acc"])
        macc_list.append(r["per_class_acc"])
        logger.info(f"  run {i+1}/{args.runs}  OA={r['overall_acc']:.2f}%  mAcc={r['per_class_acc']:.2f}%")

    oa_mean   = round(np.mean(oa_list), 2)
    oa_std    = round(np.std(oa_list),  2)
    macc_mean = round(np.mean(macc_list), 2)

    print(f"\n{'='*45}")
    print(f"  Runs             : {args.runs}")
    print(f"  Overall Accuracy : {oa_mean:.2f}% ± {oa_std:.2f}%")
    print(f"  Mean Class Acc   : {macc_mean:.2f}%")
    print(f"{'='*45}")
    print(f"\n  APES (official)  : 93.53%")
    gap = oa_mean - 93.53
    print(f"  Gap vs APES      : {gap:+.2f}%")
    print(f"\n  → Report: {oa_mean:.2f}% ± {oa_std:.2f}%")


if __name__ == "__main__":
    main()
