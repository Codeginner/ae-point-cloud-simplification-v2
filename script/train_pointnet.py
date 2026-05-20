"""
script/train_pointnet.py  (T-Net version)
=========================================
Train PointNet + T-Net di ModelNet40 — target ~89%.
Checkpoint dipakai sebagai task network (frozen) di evaluate_apes_protocol.py.

Usage:
    python script/train_pointnet.py \
        --data_root ./data/modelnet40_ply_hdf5_2048 \
        --format hdf5 \
        --save_path ./checkpoints/pointnet_cls_mn40.pth
"""

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset (HDF5)
# ---------------------------------------------------------------------------

class ModelNet40H5(Dataset):
    def __init__(self, data_root, split="train", n_points=1024, augment=True):
        import h5py
        self.n_points = n_points
        self.augment  = augment and (split == "train")

        files = sorted(glob.glob(f"{data_root}/ply_data_{split}*.h5"))
        assert files, f"Tidak ada file .h5 di {data_root} untuk split '{split}'"

        all_data, all_labels = [], []
        for f in files:
            with h5py.File(f, "r") as h:
                all_data.append(h["data"][:])
                all_labels.append(h["label"][:])

        self.data   = np.concatenate(all_data,   axis=0).astype("float32")
        self.labels = np.concatenate(all_labels, axis=0).squeeze().astype("int64")
        logger.info(f"HDF5 ModelNet40 [{split}]: {len(self.data)} samples")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        pts = self.data[i][: self.n_points].copy()
        pts -= pts.mean(axis=0)
        pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)

        if self.augment:
            # Random rotation (Y-axis) + jitter — sama dengan PointNet paper
            theta = np.random.uniform(0, 2 * np.pi)
            c, s  = np.cos(theta), np.sin(theta)
            R     = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype="float32")
            pts   = pts @ R.T
            pts  += np.clip(np.random.randn(*pts.shape) * 0.01, -0.05, 0.05).astype("float32")

        return torch.FloatTensor(pts), int(self.labels[i])


# ---------------------------------------------------------------------------
# T-Net (Spatial Transformer Network)
# ---------------------------------------------------------------------------

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k    = k
        self.conv = nn.Sequential(
            nn.Conv1d(k,   64,   1), nn.BatchNorm1d(64),   nn.ReLU(),
            nn.Conv1d(64,  128,  1), nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256,  k * k),
        )

    def forward(self, x):           # x: (B, k, N)
        B = x.size(0)
        x = self.conv(x).max(dim=-1).values    # (B, 1024)
        x = self.fc(x).view(B, self.k, self.k)
        x += torch.eye(self.k, device=x.device).unsqueeze(0)
        return x                    # (B, k, k)


# ---------------------------------------------------------------------------
# PointNet + T-Net
# ---------------------------------------------------------------------------

class PointNetCls(nn.Module):
    def __init__(self, num_class=40):
        super().__init__()
        self.tnet3  = TNet(k=3)
        self.tnet64 = TNet(k=64)

        self.conv1 = nn.Sequential(nn.Conv1d(3,   64,   1), nn.BatchNorm1d(64),   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64,  64,   1), nn.BatchNorm1d(64),   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64,  128,  1), nn.BatchNorm1d(128),  nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_class),
        )

    def forward(self, x):           # x: (B, N, 3)
        x = x.permute(0, 2, 1)     # (B, 3, N)

        t3 = self.tnet3(x)          # (B, 3, 3)
        x  = torch.bmm(t3, x)      # (B, 3, N)
        x  = self.conv1(x)
        x  = self.conv2(x)          # (B, 64, N)

        t64 = self.tnet64(x)        # (B, 64, 64)
        x   = torch.bmm(t64, x)    # (B, 64, N)
        self._t64 = t64             # simpan untuk reg loss

        x = self.conv3(x)           # (B, 128, N)
        x = self.conv4(x)           # (B, 1024, N)
        x = x.max(dim=-1).values    # (B, 1024)
        return self.fc(x)

    def reg_loss(self, lam=1e-3):
        """Orthogonality regularization untuk feature T-Net."""
        t = self._t64
        B, k, _ = t.shape
        I    = torch.eye(k, device=t.device).unsqueeze(0)
        diff = torch.bmm(t, t.permute(0, 2, 1)) - I
        return lam * (diff ** 2).sum() / B


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root",   required=True,
                   help="Path ke folder dataset. Untuk --format hdf5: path ke modelnet40_ply_hdf5_2048/")
    p.add_argument("--format",      default="hdf5", choices=["hdf5"],
                   help="Format dataset (hdf5 only)")
    p.add_argument("--save_path",   default="./checkpoints/pointnet_cls_mn40.pth")
    p.add_argument("--n_points",    type=int,   default=1024)
    p.add_argument("--epochs",      type=int,   default=250)
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
    logger.info(f"Device: {device}  |  format: {args.format}")

    train_ds = ModelNet40H5(args.data_root, "train", args.n_points, augment=True)
    test_ds  = ModelNet40H5(args.data_root, "test",  args.n_points, augment=False)

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
        model.train()
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pts)
            loss   = criterion(logits, labels) + model.reg_loss()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for pts, labels in test_loader:
                pts, labels = pts.to(device), labels.to(device)
                correct += (model(pts).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        oa = correct / total * 100

        logger.info(f"[{epoch+1:3d}/{args.epochs}]  OA={oa:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}")

        if oa > best_oa:
            best_oa = oa
            torch.save({"model": model.state_dict(), "epoch": epoch, "oa": oa},
                       args.save_path)
            logger.info(f"  → Saved best  OA={best_oa:.2f}%  →  {args.save_path}")

    logger.info(f"\nDone. Best OA: {best_oa:.2f}%  (target ~89%)")


if __name__ == "__main__":
    main()
