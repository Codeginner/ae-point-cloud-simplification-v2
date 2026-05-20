"""
evaluate_apes_protocol.py
=========================
Evaluasi mengikuti protokol APES Table 8:
    Input N=1024  →  Simplifier (model lo)  →  P_simplified (M points)
                  →  PointNet pretrained (frozen)  →  OA

Hasilnya bisa langsung dibandingkan dengan tabel APES, SampleNet, S-NET, dll.

Usage — satu nilai M:
    python evaluate_apes_protocol.py \\
        --checkpoint   ./checkpoints/best.pth \\
        --pointnet_ckpt ./pointnet_cls_mn40.pth \\
        --data_root    ./data/modelnet40_ply_hdf5_2048 \\
        --M            512

Usage — sweep semua M sekaligus (repro Table 8):
    python evaluate_apes_protocol.py \\
        --checkpoint   ./checkpoints/best.pth \\
        --pointnet_ckpt ./pointnet_cls_mn40.pth \\
        --data_root    ./data/modelnet40_ply_hdf5_2048 \\
        --M_list 512 256 128 64 32

Cara dapat pretrained PointNet:
    # Opsi A — clone SampleNet, ambil checkpoint-nya
    git clone https://github.com/itailang/SampleNet
    # checkpoint ada di: SampleNet/classification/log/pointnet/

    # Opsi B — train sendiri (~30 menit GPU)
    python script/train_pointnet.py --data_root ./data/modelnet40_ply_hdf5_2048
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposed_method.model import PointCloudSimplifier
from proposed_method.train import PointCloudDataset, DATASET_CONFIG, SUPPORTED_DATASETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PointNet — task network (frozen), arsitektur standar yang dipakai APES
# ---------------------------------------------------------------------------

class PointNetCls(nn.Module):
    """PointNet classifier standar untuk ModelNet40.

    Arsitektur ini identik dengan yang dipakai APES, SampleNet, S-NET, LighTN
    sebagai task network — sehingga angka OA bisa dibandingkan langsung.
    """

    def __init__(self, num_class: int = 40) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) → logits: (B, num_class)"""
        x = x.permute(0, 2, 1)                    # (B, 3, N)
        x = self.conv3(self.conv2(self.conv1(x)))  # (B, 1024, N)
        x = x.max(dim=-1).values                   # (B, 1024) global max-pool
        return self.fc(x)                          # (B, num_class)


# ---------------------------------------------------------------------------
# Dataset — pakai PointCloudDataset yang sudah ada (format .npy)
# ---------------------------------------------------------------------------
# PointCloudDataset membaca dari:
#   data_root/{dataset}/pcd/{train,test}/*.npy   — (2048, 3)
#   data_root/{dataset}/label/{train,test}/*.npy — scalar label
# Ini adalah format yang dihasilkan download_modelnet40.py di repo ini.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one_M(
    simplifier: PointCloudSimplifier,
    pointnet:   PointNetCls,
    loader:     DataLoader,
    M:          int,
    device:     torch.device,
    tta:        bool = False,
    tta_runs:   int  = 10,
) -> float:
    """Return Overall Accuracy (%) untuk sampling size M."""
    simplifier.eval()
    pointnet.eval()

    correct = total = 0

    for pts, labels in loader:
        pts    = pts.to(device)       # (B, N, 3)
        labels = labels.to(device)   # (B,)

        if tta:
            logits_sum = None
            for _ in range(tta_runs):
                theta  = torch.rand(1).item() * 2 * 3.14159
                cos_t  = torch.cos(torch.tensor(theta))
                sin_t  = torch.sin(torch.tensor(theta))
                rot    = torch.tensor(
                    [[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]],
                    device=device, dtype=torch.float32,
                )
                pts_aug = pts @ rot.T + torch.randn_like(pts) * 0.01
                out     = simplifier(pts_aug, compute_loss=False)
                logits  = pointnet(out["P_simplified"])
                logits_sum = logits if logits_sum is None else logits_sum + logits
            preds = logits_sum.argmax(dim=1)
        else:
            out   = simplifier(pts, compute_loss=False)
            P_s   = out["P_simplified"]     # (B, M, 3)
            preds = pointnet(P_s).argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total * 100.0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluasi APES Table 8 protocol — PointNet frozen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Path ──────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",    required=True,
                   help="Path ke checkpoint simplifier lo (best.pth)")
    p.add_argument("--pointnet_ckpt", required=True,
                   help="Path ke pretrained PointNet checkpoint (.pth)")
    p.add_argument("--data_root",     default="./data",
                   help="Root folder data (berisi sub-folder modelnet40/ atau modelnet10/)")
    p.add_argument("--dataset",       default="modelnet40", choices=SUPPORTED_DATASETS,
                   help="Dataset yang dipakai")

    # ── Model ─────────────────────────────────────────────────────────
    p.add_argument("--M",      type=int, default=None,
                   help="Satu nilai M untuk dievaluasi (gunakan ini ATAU --M_list)")
    p.add_argument("--M_list", type=int, nargs="+", default=None,
                   help="Sweep beberapa M sekaligus, contoh: --M_list 512 256 128 64 32")
    p.add_argument("--n_points",   type=int, default=1024,
                   help="Jumlah input points ke simplifier")
    p.add_argument("--num_class",  type=int, default=40,
                   help="Jumlah kelas PointNet (40 untuk ModelNet40)")

    # ── Eval ──────────────────────────────────────────────────────────
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tta",         action="store_true",
                   help="Aktifkan Test-Time Augmentation")
    p.add_argument("--tta_runs",    type=int, default=10,
                   help="Jumlah augmentation per sampel saat TTA")
    p.add_argument("--seed",        type=int, default=42)

    args = p.parse_args()

    # Validasi: harus ada M atau M_list
    if args.M is None and args.M_list is None:
        p.error("Harus set salah satu: --M atau --M_list")
    if args.M is not None and args.M_list is not None:
        p.error("Pilih salah satu saja: --M atau --M_list, bukan keduanya")
    if args.M is not None:
        args.M_list = [args.M]

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  TTA: {args.tta}  |  dataset: {args.dataset}")

    # ── Dataset — pakai PointCloudDataset (format .npy) ──────────────
    cfg = DATASET_CONFIG[args.dataset]
    if args.num_class == 40 and cfg["num_class"] != 40:
        # auto-correct kalau user lupa ganti --num_class
        args.num_class = cfg["num_class"]
    test_ds = PointCloudDataset(
        data_root=args.data_root, mode="test",
        n_points=args.n_points, augment=False,
        dataset=args.dataset,
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ── PointNet (frozen) ─────────────────────────────────────────────
    pointnet = PointNetCls(num_class=args.num_class).to(device)
    pn_ckpt  = torch.load(args.pointnet_ckpt, map_location=device)
    # handle berbagai format checkpoint PointNet
    pn_state = pn_ckpt.get("model_state_dict", pn_ckpt.get("model", pn_ckpt))
    pointnet.load_state_dict(pn_state)
    pointnet.eval()
    for p in pointnet.parameters():
        p.requires_grad_(False)
    logger.info(f"PointNet loaded: {args.pointnet_ckpt}")

    # Sanity check PointNet di full 1024 points (harusnya ~89%)
    logger.info("Sanity check PointNet dengan input penuh (N=1024)...")
    correct = total = 0
    with torch.no_grad():
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            preds = pointnet(pts).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    pn_oa = correct / total * 100
    logger.info(f"PointNet OA (N=1024, baseline): {pn_oa:.2f}%  (expect ~89%)")

    # ── Simplifier ────────────────────────────────────────────────────
    # Inisialisasi dengan M terbesar, nanti diganti per iterasi
    max_M = max(args.M_list)
    simplifier = PointCloudSimplifier(M=max_M, num_class=args.num_class).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    simplifier.load_state_dict(state)
    logger.info(f"Simplifier loaded: {args.checkpoint}")

    # ── Sweep M ───────────────────────────────────────────────────────
    results = {}
    for M in args.M_list:
        logger.info(f"Evaluasi M={M}...")

        # Update M di semua submodule yang memakainya
        simplifier.M = M
        if hasattr(simplifier, "selector"):
            simplifier.selector.M = M
        if hasattr(simplifier, "decoder") and hasattr(simplifier.decoder, "M"):
            simplifier.decoder.M = M

        oa = evaluate_one_M(
            simplifier, pointnet, loader, M, device,
            tta=args.tta, tta_runs=args.tta_runs,
        )
        results[M] = oa
        logger.info(f"  M={M:4d}  OA={oa:.2f}%")

    # ── Tabel hasil ───────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Protokol: APES Table 8  |  Task network: PointNet")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  TTA: {args.tta}")
    print(f"{'='*52}")
    print(f"  {'M':>6}  {'OA (%)':>10}  {'vs APES (global)':>18}")
    print(f"  {'-'*40}")

    # Referensi APES (global) dari Table 8
    apes_ref = {512: 90.81, 256: 90.40, 128: 89.77, 64: None, 32: None}

    for M, oa in sorted(results.items(), reverse=True):
        ref  = apes_ref.get(M)
        gap  = f"{oa - ref:+.2f}%" if ref is not None else "   N/A"
        print(f"  {M:>6}  {oa:>10.2f}%  {gap:>18}")

    print(f"{'='*52}\n")
    print("Paste angka di atas ke skripsi/paper lo sebagai:")
    print("  Table X — OA ModelNet40 protokol APES, PointNet task network\n")


if __name__ == "__main__":
    main()
