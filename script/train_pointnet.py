"""
script/train_pointnet.py
========================
Train PointNet classifier di ModelNet40 (~30 menit di 1 GPU).
Hasilnya dipakai sebagai task network (frozen) di evaluate_apes_protocol.py.

Kompatibel dengan format .npy hasil download_modelnet40.py:
    data_root/modelnet40/pcd/{train,test}/*.npy
    data_root/modelnet40/label/{train,test}/*.npy

Usage:
    python script/train_pointnet.py \\
        --data_root /kaggle/working/ae-point-cloud-simplification-v2/data \\
        --save_path ./checkpoints/pointnet_cls_mn40.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Reuse PointCloudDataset yang sudah ada di repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proposed_method.train import PointCloudDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PointNet — arsitektur identik dengan evaluate_apes_protocol.py
# ---------------------------------------------------------------------------

class PointNetCls(nn.Module):
    def __init__(self, num_class: int = 40) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) → logits: (B, num_class)"""
        x = x.permute(0, 2, 1)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.max(dim=-1).values
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root",   default="./data",
                   help="Root folder data — berisi sub-folder modelnet40/")
    p.add_argument("--dataset",     default="modelnet40", choices=["modelnet40", "modelnet10"],
                   help="Dataset yang dipakai")
    p.add_argument("--save_path",   default="./checkpoints/pointnet_cls_mn40.pth")
    p.add_argument("--n_points",    type=int,   default=1024)
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  dataset: {args.dataset}")

    # ── Data — pakai PointCloudDataset yang sudah ada ─────────────────
    train_ds = PointCloudDataset(args.data_root, mode="train",
                                 n_points=args.n_points, augment=True,
                                 dataset=args.dataset)
    test_ds  = PointCloudDataset(args.data_root, mode="test",
                                 n_points=args.n_points, augment=False,
                                 dataset=args.dataset)
    num_class = train_ds.num_class

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}  Test: {len(test_ds)}  num_class: {num_class}")

    # ── Model ─────────────────────────────────────────────────────────
    model     = PointNetCls(num_class=num_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    best_oa = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pts), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for pts, labels in test_loader:
                pts, labels = pts.to(device), labels.to(device)
                correct += (model(pts).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        oa = correct / total * 100

        logger.info(
            f"[{epoch+1:3d}/{args.epochs}]  "
            f"loss={train_loss/len(train_loader):.4f}  "
            f"OA={oa:.2f}%  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if oa > best_oa:
            best_oa = oa
            torch.save({"model": model.state_dict(), "epoch": epoch, "oa": oa},
                       args.save_path)
            logger.info(f"  → Saved best  OA={best_oa:.2f}%  →  {args.save_path}")

    logger.info(f"\nDone. Best OA: {best_oa:.2f}%  (target ~89%)")


if __name__ == "__main__":
    main()
