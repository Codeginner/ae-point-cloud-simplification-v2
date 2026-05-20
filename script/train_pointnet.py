"""
script/train_pointnet.py
========================
Train PointNet classifier di ModelNet40 (~30 menit di 1 GPU).
Hasilnya dipakai sebagai task network (frozen) di evaluate_apes_protocol.py
dan train_ddp.py (--pointnet_ckpt).

Support dua format dataset:
  --format hdf5  : modelnet40_ply_hdf5_2048  (official, target ~89%)
  --format npy   : modelnet40/pcd/           (format repo ini)

Usage (HDF5 — recommended):
    python script/train_pointnet.py \\
        --data_root ./data/modelnet40_ply_hdf5_2048 \\
        --format hdf5 \\
        --save_path ./checkpoints/pointnet_cls_mn40.pth

Usage (npy):
    python script/train_pointnet.py \\
        --data_root ./data \\
        --format npy \\
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
# HDF5 Dataset (format official APES / SampleNet)
# ---------------------------------------------------------------------------

class ModelNet40H5(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 n_points: int = 1024, augment: bool = True) -> None:
        import h5py
        self.n_points = n_points
        self.augment  = augment and (split == "train")

        files = sorted(glob.glob(f"{data_root}/ply_data_{split}*.h5"))
        assert files, (
            f"Tidak ada file .h5 di {data_root} untuk split '{split}'. "
            "Pastikan path ke modelnet40_ply_hdf5_2048."
        )

        all_data, all_labels = [], []
        for f in files:
            with h5py.File(f, "r") as h:
                all_data.append(h["data"][:])
                all_labels.append(h["label"][:])

        self.data   = np.concatenate(all_data,   axis=0).astype("float32")  # (N, 2048, 3)
        self.labels = np.concatenate(all_labels, axis=0).squeeze().astype("int64")
        logger.info(f"HDF5 ModelNet40 [{split}]: {len(self.data)} samples")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        pts = self.data[i][: self.n_points].copy()   # (1024, 3)

        # Normalisasi: zero-mean + unit sphere
        pts -= pts.mean(axis=0)
        pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)

        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s  = np.cos(theta), np.sin(theta)
            R     = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype="float32")
            pts   = pts @ R.T
            pts  += np.clip(np.random.randn(*pts.shape) * 0.01, -0.05, 0.05).astype("float32")

        return torch.FloatTensor(pts), int(self.labels[i])


# ---------------------------------------------------------------------------
# NPY Dataset (format repo ini)
# ---------------------------------------------------------------------------

class ModelNet40NPY(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 n_points: int = 1024, augment: bool = True) -> None:
        from proposed_method.train import PointCloudDataset
        self._ds = PointCloudDataset(data_root, mode=split,
                                     n_points=n_points, augment=augment,
                                     dataset="modelnet40")

    def __len__(self): return len(self._ds)
    def __getitem__(self, i): return self._ds[i]


# ---------------------------------------------------------------------------
# PointNet — arsitektur identik dengan evaluate_apes_protocol.py
# ---------------------------------------------------------------------------

class PointNetCls(nn.Module):
    def __init__(self, num_class: int = 40) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3,64,1),    nn.BatchNorm1d(64),   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64,128,1),  nn.BatchNorm1d(128),  nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_class),
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.max(dim=-1).values
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root",   required=True,
                   help="Path ke folder dataset. "
                        "Untuk --format hdf5: path ke modelnet40_ply_hdf5_2048/. "
                        "Untuk --format npy:  path ke root data/ (berisi modelnet40/).")
    p.add_argument("--format",      default="hdf5", choices=["hdf5", "npy"],
                   help="Format dataset: 'hdf5' (official) atau 'npy' (repo ini)")
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
    logger.info(f"Device: {device}  |  format: {args.format}")

    # ── Dataset ───────────────────────────────────────────────────────
    DS = ModelNet40H5 if args.format == "hdf5" else ModelNet40NPY
    train_ds = DS(args.data_root, "train", args.n_points, augment=True)
    test_ds  = DS(args.data_root, "test",  args.n_points, augment=False)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
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
            criterion(model(pts), labels).backward()
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

        logger.info(f"[{epoch+1:3d}/{args.epochs}]  OA={oa:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}")

        if oa > best_oa:
            best_oa = oa
            torch.save({"model": model.state_dict(), "epoch": epoch, "oa": oa},
                       args.save_path)
            logger.info(f"  → Saved best  OA={best_oa:.2f}%  →  {args.save_path}")

    logger.info(f"\nDone. Best OA: {best_oa:.2f}%  (target ~89%)")


if __name__ == "__main__":
    main()


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
