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
def evaluate(model, loader, device, num_class):
    model.eval()
    all_preds, all_labels = [], []

    for P, labels in loader:
        P      = P.to(device)
        labels = labels.to(device)

        out    = model(P, labels=None, compute_loss=False)
        preds  = out["logits"].argmax(dim=-1)

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
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

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

    # Evaluate
    result = evaluate(model, loader, device, args.num_class)

    print(f"\n{'='*40}")
    print(f"  Overall Accuracy : {result['overall_acc']:.2f}%")
    print(f"  Mean Class Acc   : {result['per_class_acc']:.2f}%")
    print(f"{'='*40}")
    print(f"\n  APES (official)  : 93.53%")
    gap = result['overall_acc'] - 93.53
    print(f"  Gap vs APES      : {gap:+.2f}%")


if __name__ == "__main__":
    main()
