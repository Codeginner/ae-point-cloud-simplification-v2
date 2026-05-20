"""
script/train_pointnet.py
========================
Train PointNet classifier di ModelNet40 (~30 menit di 1 GPU).
Hasilnya dipakai sebagai task network (frozen) di evaluate_apes_protocol.py.

Usage:
    python script/train_pointnet.py \\
        --data_root ./data/modelnet40_ply_hdf5_2048 \\
        --save_path ./checkpoints/pointnet_cls_mn40.pth
"""

import argparse
import glob
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset (reuse format yang sama dengan evaluate_apes_protocol.py)
# ---------------------------------------------------------------------------

class ModelNet40H5(Dataset):
    def __init__(self, data_root: str, split: str = "train", n_points: int = 1024,
                 augment: bool = True) -> None:
        self.n_points = n_points
        self.augment  = augment and (split == "train")

        files = sorted(glob.glob(f"{data_root}/ply_data_{split}*.h5"))
        assert files, f"Tidak ada HDF5 di {data_root} untuk split '{split}'"

        all_data, all_labels = [], []
        for f in files:
            with h5py.File(f, "r") as h:
                all_data.append(h["data"][:])
                all_labels.append(h["label"][:])

        self.data   = np.concatenate(all_data,   axis=0).astype("float32")
        self.labels = np.concatenate(all_labels, axis=0).squeeze().astype("int64")
        logger.info(f"ModelNet40 [{split}]: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        pts = self.data[i][: self.n_points].copy()

        # Normalisasi
        pts -= pts.mean(axis=0)
        pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)

        if self.augment:
            # Random rotation (Y-axis)
            theta  = np.random.uniform(0, 2 * np.pi)
            c, s   = np.cos(theta), np.sin(theta)
            R      = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype="float32")
            pts    = pts @ R.T
            # Jitter
            pts   += np.clip(np.random.randn(*pts.shape) * 0.01, -0.05, 0.05).astype("float32")

        return torch.FloatTensor(pts), int(self.labels[i])


# ---------------------------------------------------------------------------
# PointNet (arsitektur identik dengan evaluate_apes_protocol.py)
# ---------------------------------------------------------------------------

class PointNetCls(nn.Module):
    def __init__(self, num_class: int = 40) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128),  nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.max(dim=-1).values
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root",  default="./data/modelnet40_ply_hdf5_2048")
    p.add_argument("--save_path",  default="./checkpoints/pointnet_cls_mn40.pth",
                   help="Path untuk menyimpan checkpoint PointNet")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ds = ModelNet40H5(args.data_root, "train", augment=True)
    test_ds  = ModelNet40H5(args.data_root, "test",  augment=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model     = PointNetCls(num_class=40).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    best_oa = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pts), labels)
            loss.backward()
            optimizer.step()
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

        logger.info(f"[Epoch {epoch+1}/{args.epochs}]  OA={oa:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}")

        if oa > best_oa:
            best_oa = oa
            torch.save({"model": model.state_dict(), "epoch": epoch, "oa": oa},
                       args.save_path)
            logger.info(f"  → Saved best PointNet  OA={best_oa:.2f}%")

    logger.info(f"\nDone. Best OA: {best_oa:.2f}%  (expect ~89%)")
    logger.info(f"Checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
