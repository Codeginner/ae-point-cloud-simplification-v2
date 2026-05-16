"""
train.py — Training loop for PointCloudSimplifier.

Usage:
    python -m proposed_method.train --data_root /path/to/dataset
"""
import os
import argparse
import logging
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from .model import PointCloudSimplifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# ModelNet10 Dataset
# ---------------------------------------------------------------------------

SUBSET_CLASSES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
    'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]

class PointCloudDataset(Dataset):
    """ModelNet10 subset dataset.

    Membaca file .npy hasil download_modelnet10.py dengan struktur:
        data_root/modelnet10/pcd/{mode}/0000.npy    — (2048, 3)
        data_root/modelnet10/label/{mode}/0000.npy  — scalar 0-9

    Args:
        data_root:  Root folder data, default './data'.
        mode:       'train' atau 'test'.
        n_points:   Jumlah point yang dipakai per sampel. Jika < 2048,
                    dilakukan random sampling. Default 1024.
        augment:    Aktifkan augmentasi (random rotation + jitter)
                    saat training. Default True.
    """

    def __init__(
        self,
        data_root: str  = './data',
        mode:      str  = 'train',
        n_points:  int  = 1024,
        augment:   bool = True,
    ) -> None:
        super().__init__()
        assert mode in ('train', 'test'), "mode harus 'train' atau 'test'"
        self.n_points = n_points
        self.augment  = augment and (mode == 'train')

        import glob
        pcd_dir   = os.path.join(data_root, 'modelnet10', 'pcd',   mode)
        label_dir = os.path.join(data_root, 'modelnet10', 'label', mode)

        self.pcd_files   = sorted(glob.glob(os.path.join(pcd_dir,   '*.npy')))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))

        assert len(self.pcd_files) > 0, \
            f"Tidak ada file di {pcd_dir}. Jalankan download_modelnet10.py dulu."
        assert len(self.pcd_files) == len(self.label_files), \
            "Jumlah file pcd dan label tidak sama."

        logger.info(f"ModelNet10 [{mode}]: {len(self.pcd_files)} samples, "
                    f"n_points={n_points}, augment={self.augment}")

    def __len__(self) -> int:
        return len(self.pcd_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        # Load
        pcd   = np.load(self.pcd_files[idx]).astype('float32')   # (2048, 3)
        label = np.load(self.label_files[idx])                    # scalar

        # Random subsample
        if self.n_points < len(pcd):
            choice = np.random.choice(len(pcd), self.n_points, replace=False)
            pcd    = pcd[choice]

        # Normalise: zero-mean + unit sphere
        pcd -= pcd.mean(axis=0)
        scale = np.max(np.linalg.norm(pcd, axis=1))
        pcd  /= (scale + 1e-8)

        pcd = torch.from_numpy(pcd)                              # (n_points, 3)

        # Augmentasi
        if self.augment:
            pcd = self._random_rotate(pcd)
            pcd = self._random_jitter(pcd)

        return pcd, torch.tensor(int(label), dtype=torch.long)

    # ------------------------------------------------------------------
    # Augmentasi
    # ------------------------------------------------------------------

    def _random_rotate(self, pcd: torch.Tensor) -> torch.Tensor:
        """Rotasi random di sumbu Y (up-axis)."""
        theta  = torch.rand(1) * 2 * torch.pi
        cos_t, sin_t = theta.cos(), theta.sin()
        R = torch.tensor([
            [ cos_t, 0, sin_t],
            [     0, 1,     0],
            [-sin_t, 0, cos_t],
        ], dtype=torch.float32).squeeze()
        return pcd @ R.T

    def _random_jitter(self, pcd: torch.Tensor, sigma: float = 0.01, clip: float = 0.05) -> torch.Tensor:
        """Tambahkan Gaussian noise kecil ke setiap point."""
        noise = torch.clamp(torch.randn_like(pcd) * sigma, -clip, clip)
        return pcd + noise


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      PointCloudSimplifier,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"total": 0.0, "chamfer": 0.0, "normal": 0.0, "nc": 0.0}

    for step, (P, _) in enumerate(loader):          # unpack (pcd, label); label tidak dipakai
        P = P.to(device)                             # (B, N, 3)

        optimizer.zero_grad()

        out  = model(P, compute_loss=True)
        loss = out["loss"]

        # .mean() untuk handle DataParallel yang return tensor per-GPU
        total_loss = loss["total"].mean()
        total_loss.backward()
        optimizer.step()

        for k, v in loss.items():
            totals[k] += v.mean().item()

        if step % 50 == 0:
            logger.info(
                f"Epoch {epoch}  step {step}/{len(loader)}  "
                f"loss={loss['total'].mean().item():.4f}  "
                f"cd={loss['chamfer'].mean().item():.4f}  "
                f"n={loss['normal'].mean().item():.4f}  "
                f"nc={loss['nc'].mean().item():.4f}"
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:  PointCloudSimplifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"total": 0.0, "chamfer": 0.0, "normal": 0.0, "nc": 0.0}

    for P, _ in loader:                              # unpack (pcd, label)
        P    = P.to(device)
        out  = model(P, compute_loss=True)
        loss = out["loss"]
        for k, v in loss.items():
            totals[k] += v.mean().item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}

# ------------------------------------------------------------------
# visualize 3d point cloud
# ------------------------------------------------------------------

@torch.no_grad()
def visualize_results(
    model: PointCloudSimplifier,
    loader: DataLoader,
    device: torch.device,
    save_dir: str = "./visualizations",
    num_samples: int = 3,
) -> None:
    """Visualize simplification results."""

    model.eval()

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(loader):

        if idx >= num_samples:
            break

        # BUG FIX: batch is a (pcd, label) tuple — cannot call .to() on a tuple
        P, _ = batch
        P = P.to(device)

        out = model(P, compute_loss=False)

        P_original = P[0].cpu().numpy()
        P_simple   = out["P_simplified"][0].cpu().numpy()
        P_recon    = out["P_recon"][0].cpu().numpy() if "P_recon" in out else None

        # BUG FIX: visualize_point_clouds was not imported; use matplotlib directly
        fig = plt.figure(figsize=(15, 5))
        titles  = ["Original", "Simplified", "Reconstructed"]
        clouds  = [P_original, P_simple, P_recon]
        for col, (title, pts) in enumerate(zip(titles, clouds)):
            ax = fig.add_subplot(1, 3, col + 1, projection="3d")
            if pts is not None:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
            ax.set_title(title)
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointCloudSimplifier")
    parser.add_argument("--data_root",    type=str,   default="./data",
                        help="Root folder data (berisi modelnet10/)")
    parser.add_argument("--n_points",     type=int,   default=1024,
                        help="Jumlah point per sampel")
    parser.add_argument("--M",            type=int,   default=512,
                        help="Jumlah output simplified points (harus <= n_points)")
    parser.add_argument("--k",            type=int,   default=20,
                        help="KNN neighbours")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path ke checkpoint untuk resume training")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  GPU count: {torch.cuda.device_count()}")

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = PointCloudDataset(data_root=args.data_root, mode='train',
                                 n_points=args.n_points, augment=True)
    val_ds   = PointCloudDataset(data_root=args.data_root, mode='test',
                                 n_points=args.n_points, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────
    model = PointCloudSimplifier(M=args.M, k=args.k)

    if torch.cuda.device_count() > 1:
        logger.info(f"Pakai {torch.cuda.device_count()} GPU via DataParallel")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        # handle DataParallel wrapper
        target = model.module if hasattr(model, 'module') else model
        target.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_losses   = validate(model, val_loader, device)

        if epoch % 10 == 0:
            visualize_results(
                model=model,
                loader=val_loader,
                device=device,
                save_dir="./visualizations",
                num_samples=1,
            )

        scheduler.step()

        logger.info(
            f"[Epoch {epoch}]  "
            f"train={train_losses['total']:.4f}  "
            f"val={val_losses['total']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save checkpoint — akses .module kalau DataParallel
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        torch.save({
            "epoch":     epoch,
            "model":     state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss":  val_losses,
        }, ckpt_dir / "latest.pth")

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(state_dict, ckpt_dir / "best.pth")
            logger.info(f"  → New best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()