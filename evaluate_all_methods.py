"""
evaluate_all_methods.py
=======================
Evaluasi semua metode simplifikasi dengan frozen PointNet task network.
Protokol identik dengan APES Table 8 sehingga hasilnya bisa langsung
dibandingkan dengan angka published di paper.

Methods evaluated:
    ── Baselines ──────────────────────────────────────────
    • Full Cloud    : PointNet langsung pada N=1024 (upper bound)
    • Random        : random sampling
    • FPS           : Farthest Point Sampling
    • Voxel         : voxel grid downsampling
    ── SOTA (published numbers, tidak dijalankan) ─────────
    • S-NET         : Dovrat et al. 2019
    • SampleNet     : Lang et al. 2020
    • APES          : Kim et al. 2023
    ── Proposed ───────────────────────────────────────────
    • Ours          : PointCloudSimplifier (checkpoint lo)

Output:
    • Tabel ASCII di terminal
    • eval_results/pointnet_protocol_results.json
    • eval_results/pointnet_protocol_table.png  (publication-ready figure)

Usage:
    python evaluate_all_methods.py \\
        --checkpoint    ./checkpoints/best.pth \\
        --pointnet_ckpt ./checkpoints/pointnet_cls_mn40.pth \\
        --data_root     ./data \\
        --M_list 512 256 128 64 32 \\
        --dataset       modelnet40 \\
        --out_dir       ./eval_results

Catatan protokol:
    • PointNet ditraining pada N=1024 penuh, lalu di-freeze
    • Semua metode simplifikasi dijalankan pada test set (no retraining)
    • Angka SOTA diambil langsung dari paper masing-masing:
        S-NET   : Dovrat et al. (2019) Table 2
        SampleNet: Lang et al. (2020) Table 1
        APES    : Kim et al. (2023) Table 8
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposed_method.model import PointCloudSimplifier
from proposed_method.train import (
    PointCloudDataset, ModelNet40H5, build_dataset,
    DATASET_CONFIG, SUPPORTED_DATASETS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# SOTA Published Numbers  (PointNet task network, ModelNet40)
# Source: respective papers, evaluation on full test set
# ===========================================================================

# Format: { M: OA% }
# None = not reported in that paper
SOTA_PUBLISHED = {
    "S-NET": {          # Dovrat et al. 2019, ICCV
        # "Pruning and Simplifying Point Sets for Neural Networks"
        # Table 2, PointNet classifier, ModelNet40
        1024: 87.7,
        512:  87.4,
        256:  86.7,
        128:  84.0,
        64:   79.0,
        32:   None,
    },
    "SampleNet": {      # Lang et al. 2020, CVPR
        # "SampleNet: Differentiable Point Cloud Sampling"
        # Table 1, PointNet classifier, ModelNet40
        1024: 87.7,
        512:  87.4,
        256:  87.0,
        128:  85.0,
        64:   80.7,
        32:   None,
    },
    "APES": {           # Kim et al. 2023, CVPR
        # "APES: Attention-based Point cloud Edge Sampling"
        # Table 8, PointNet classifier, ModelNet40
        1024: 89.2,
        512:  88.8,
        256:  88.2,
        128:  87.2,
        64:   84.4,
        32:   79.9,
    },
}


# ===========================================================================
# PointNet (frozen task network)
# ===========================================================================

class TNet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
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

    def forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x = self.conv(x).max(dim=-1).values
        x = self.fc(x).view(B, self.k, self.k)
        x += torch.eye(self.k, device=x.device).unsqueeze(0)
        return x


class PointNetCls(nn.Module):
    def __init__(self, num_class: int = 40):
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

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, N, 3) → logits: (B, num_class)"""
        x = x.permute(0, 2, 1)
        t3 = self.tnet3(x);  x = torch.bmm(t3, x)
        x  = self.conv1(x);  x = self.conv2(x)
        t64 = self.tnet64(x); x = torch.bmm(t64, x)
        x = self.conv3(x);   x = self.conv4(x)
        x = x.max(dim=-1).values
        return self.fc(x)


def load_pointnet(ckpt_path: str, num_class: int, device: torch.device) -> PointNetCls:
    pn = PointNetCls(num_class=num_class).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    pn.load_state_dict(state)
    pn.eval()
    for p in pn.parameters():
        p.requires_grad_(False)
    logger.info(f"PointNet loaded + frozen: {ckpt_path}")
    return pn


# ===========================================================================
# Baseline Sampling Methods
# ===========================================================================

def random_sample(P: Tensor, M: int) -> Tensor:
    B, N, _ = P.shape
    idx = torch.stack([torch.randperm(N, device=P.device)[:M] for _ in range(B)])
    return P.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps(P: Tensor, M: int) -> Tensor:
    B, N, _ = P.shape
    device   = P.device
    selected = torch.zeros(B, M, dtype=torch.long, device=device)
    dist     = torch.full((B, N), float("inf"), device=device)
    selected[:, 0] = torch.randint(N, (B,), device=device)
    for i in range(1, M):
        last = P.gather(1, selected[:, i-1:i].unsqueeze(-1).expand(-1, -1, 3))
        d    = ((P - last) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        selected[:, i] = dist.argmax(dim=1)
    return P.gather(1, selected.unsqueeze(-1).expand(-1, -1, 3))


def voxel_downsample(P: Tensor, M: int) -> Tensor:
    B, N, C = P.shape
    results = []
    for b in range(B):
        pts  = P[b]
        mn   = pts.min(0).values
        mx   = pts.max(0).values
        span = (mx - mn).max().item()
        lo, hi = span / N, span
        for _ in range(20):
            mid    = (lo + hi) / 2
            voxels = ((pts - mn) / mid).long()
            unique = torch.unique(voxels, dim=0)
            if unique.shape[0] >= M: hi = mid
            else:                    lo = mid
            if abs(unique.shape[0] - M) <= 2: break
        voxel_ids = ((pts - mn) / hi).long()
        key = voxel_ids[:, 0] * 100003 + voxel_ids[:, 1] * 1003 + voxel_ids[:, 2]
        _, inv = torch.unique(key, return_inverse=True)
        n_vox  = inv.max().item() + 1
        centres = torch.zeros(n_vox, 3, device=pts.device)
        counts  = torch.zeros(n_vox, 1, device=pts.device)
        centres.scatter_add_(0, inv.unsqueeze(-1).expand(-1, 3), pts)
        counts.scatter_add_(0, inv.unsqueeze(-1), torch.ones(N, 1, device=pts.device))
        centres = centres / counts.clamp(1)
        if centres.shape[0] >= M:
            idx = torch.randperm(centres.shape[0], device=pts.device)[:M]
            results.append(centres[idx])
        else:
            rep = M - centres.shape[0]
            pad = centres[torch.randint(centres.shape[0], (rep,), device=pts.device)]
            results.append(torch.cat([centres, pad], 0))
    return torch.stack(results)


# ===========================================================================
# Evaluation runner
# ===========================================================================

@torch.no_grad()
def evaluate_method(
    pointnet:    PointNetCls,
    loader:      DataLoader,
    simplify_fn,           # callable(P: Tensor, M: int) → (B, M, 3)
    M:           int,
    device:      torch.device,
    num_class:   int,
    label:       str = "",
) -> dict:
    """Evaluate one (method, M) pair. Returns OA, per-class acc, timing."""
    pointnet.eval()
    all_preds, all_labels, times = [], [], []

    for pts, labels in loader:
        pts    = pts.to(device)
        labels = labels.to(device)

        t0  = time.perf_counter()
        P_s = simplify_fn(pts, M)          # (B, M, 3)
        elapsed_ms = (time.perf_counter() - t0) / pts.shape[0] * 1000
        times.append(elapsed_ms)

        preds = pointnet(P_s).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    oa = (preds == labels).float().mean().item() * 100

    per_class = []
    for c in range(num_class):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append((preds[mask] == c).float().mean().item())

    return {
        "oa":         round(oa, 2),
        "macc":       round(np.mean(per_class) * 100, 2),
        "avg_time_ms": round(np.mean(times), 2),
    }


@torch.no_grad()
def evaluate_full_cloud(
    pointnet: PointNetCls,
    loader:   DataLoader,
    device:   torch.device,
    num_class: int,
) -> dict:
    """Baseline: PointNet pada full N=1024 cloud."""
    pointnet.eval()
    all_preds, all_labels, times = [], [], []
    for pts, labels in loader:
        pts    = pts.to(device)
        labels = labels.to(device)
        t0     = time.perf_counter()
        preds  = pointnet(pts).argmax(dim=1)
        times.append((time.perf_counter() - t0) / pts.shape[0] * 1000)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    oa     = (preds == labels).float().mean().item() * 100
    per_class = []
    for c in range(num_class):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append((preds[mask] == c).float().mean().item())
    return {
        "oa":          round(oa, 2),
        "macc":        round(np.mean(per_class) * 100, 2),
        "avg_time_ms": round(np.mean(times), 2),
    }


# ===========================================================================
# Publication-ready figure
# ===========================================================================

# Visual style per method
STYLE = {
    # ── Proposed ──────────────────────────────────────────────────────
    "Ours":      dict(color="#E05C5C", lw=2.8, marker="*",  ms=12, zorder=10,
                      ls="-",  label="Ours (proposed)"),
    # ── Baselines ─────────────────────────────────────────────────────
    "FPS":       dict(color="#5BA3E8", lw=1.8, marker="o",  ms=7,  zorder=4,
                      ls="-",  label="FPS"),
    "Random":    dict(color="#A0A0A0", lw=1.5, marker="s",  ms=6,  zorder=3,
                      ls="--", label="Random Sampling"),
    "Voxel":     dict(color="#F0A030", lw=1.8, marker="^",  ms=7,  zorder=4,
                      ls="-",  label="Voxel Grid"),
    # ── SOTA ──────────────────────────────────────────────────────────
    "S-NET":     dict(color="#B0B0FF", lw=1.8, marker="D",  ms=6,  zorder=5,
                      ls=":",  label="S-NET (Dovrat et al., 2019)"),
    "SampleNet": dict(color="#C07AD8", lw=2.0, marker="v",  ms=7,  zorder=6,
                      ls=":",  label="SampleNet (Lang et al., 2020)"),
    "APES":      dict(color="#4DCCA0", lw=2.2, marker="P",  ms=8,  zorder=7,
                      ls=":",  label="APES (Kim et al., 2023)"),
}

# Full-cloud reference lines
FULL_CLOUD_STYLE = dict(color="#666666", lw=1.2, ls="--", alpha=0.6,
                        label="PointNet full cloud (1024 pts)")


def make_figure(
    M_values:    list,
    run_results: dict,         # { "Ours"/"FPS"/... : { M: {"oa":..} } }
    full_cloud_oa: float,
    save_path:   Path,
) -> None:
    """OA vs M — all methods on one plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0f0f14")
    ax.set_facecolor("#0f0f14")

    # Full-cloud dashed reference
    ax.axhline(full_cloud_oa, **FULL_CLOUD_STYLE)

    # SOTA published — only plot at M values that exist
    for name, sota_dict in SOTA_PUBLISHED.items():
        xs = [m for m in M_values if sota_dict.get(m) is not None]
        ys = [sota_dict[m] for m in xs]
        if xs:
            st = STYLE[name]
            ax.plot(xs, ys, color=st["color"], lw=st["lw"],
                    marker=st["marker"], ms=st["ms"], zorder=st["zorder"],
                    ls=st["ls"], label=st["label"])

    # Computed methods
    for name, res_by_M in run_results.items():
        xs = sorted(res_by_M.keys())
        ys = [res_by_M[m]["oa"] for m in xs]
        st = STYLE.get(name, dict(color="white", lw=1.5, marker="x",
                                  ms=7, zorder=2, ls="-", label=name))
        ax.plot(xs, ys, color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=st["ms"], zorder=st["zorder"],
                ls=st["ls"], label=st["label"])

    # Axes
    ax.set_xlabel("M  (number of simplified points)", color="#aaa", fontsize=12)
    ax.set_ylabel("Overall Accuracy (%)",             color="#aaa", fontsize=12)
    ax.set_title("PointNet Task Network — OA vs Simplification Rate",
                 color="#eee", fontsize=13, pad=12, fontweight="bold")

    ax.set_xticks(M_values)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(color="#222", linestyle="--", linewidth=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    legend = ax.legend(
        facecolor="#1a1a22", edgecolor="#444",
        labelcolor="#ddd",   fontsize=9,
        loc="lower right",   framealpha=0.9,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Figure saved → {save_path}")


# ===========================================================================
# ASCII Table
# ===========================================================================

def print_table(
    M_values:       list,
    run_results:    dict,
    full_cloud_oa:  float,
    num_class:      int,
) -> None:
    all_methods = ["Full Cloud"] + list(run_results.keys()) + list(SOTA_PUBLISHED.keys())
    col_w = 10

    header  = f"{'Method':<16}" + "".join(
        f"{'M='+str(m):>{col_w}}" for m in M_values
    )
    sep = "─" * len(header)

    print(f"{sep}")
    print(f"  PointNet Downstream OA (%)  ·  ModelNet{num_class}  ·  frozen PointNet")
    print(sep)
    print(header)
    print(sep)

    # Full cloud
    print(f"  {'Full Cloud (N=1024)':<14}" +
          "".join(f"{full_cloud_oa:>{col_w}.2f}" for _ in M_values))
    print(f"  {'─'*14}")

    # Computed baselines + ours
    for name, res in run_results.items():
        row = f"  {name:<14}"
        for m in M_values:
            v = res.get(m, {}).get("oa")
            row += f"{v:>{col_w}.2f}" if v is not None else f"{'—':>{col_w}}"
        print(row)

    print(f"  {'─'*14}")

    # SOTA (published)
    for name, sota_dict in SOTA_PUBLISHED.items():
        row = f"  {name:<14}"
        for m in M_values:
            v = sota_dict.get(m)
            row += f"{v:>{col_w}.2f}*" if v is not None else f"{'—':>{col_w}}"
        print(row)

    print(sep)
    print("  * published numbers from respective papers (not re-run)")
    print(sep + "")


# ===========================================================================
# Argparse
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="All-methods eval with frozen PointNet — vs SOTA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path ke best.pth simplifier lo")
    p.add_argument("--pointnet_ckpt", required=True,
                   help="Path ke pretrained PointNet checkpoint")
    p.add_argument("--data_root",     default="./data")
    p.add_argument("--data_format",   default="npy", choices=["npy", "hdf5"])
    p.add_argument("--dataset",       default="modelnet40", choices=SUPPORTED_DATASETS)
    p.add_argument("--M_list",        nargs="+", type=int,
                   default=[512, 256, 128, 64, 32],
                   help="List of M values to sweep")
    p.add_argument("--n_points",      type=int, default=1024)
    p.add_argument("--num_class",     type=int, default=40)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--simplifier_M",  type=int, default=512,
                   help="M yang dipakai waktu training simplifier lo")
    p.add_argument("--out_dir",       default="./eval_results")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────
    test_ds = build_dataset(
        data_root=args.data_root, mode="test",
        n_points=args.n_points,   augment=False,
        dataset=args.dataset,     data_format=args.data_format,
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f"Test samples: {len(test_ds)}")

    # ── PointNet frozen ───────────────────────────────────────────────
    pointnet = load_pointnet(args.pointnet_ckpt, args.num_class, device)

    # ── Sanity check: full cloud ───────────────────────────────────────
    logger.info("Evaluating full cloud baseline (N=1024)...")
    full_res   = evaluate_full_cloud(pointnet, loader, device, args.num_class)
    full_oa    = full_res["oa"]
    logger.info(f"  Full cloud OA: {full_oa:.2f}%  (expect ~87-89% for ModelNet40)")

    # ── Our simplifier ────────────────────────────────────────────────
    logger.info(f"Loading simplifier: {args.checkpoint}")
    simplifier = PointCloudSimplifier(
        M=args.simplifier_M, num_class=args.num_class,
    ).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    simplifier.load_state_dict(state)
    simplifier.eval()

    @torch.no_grad()
    def our_fn(P: Tensor, M: int) -> Tensor:
        out = simplifier(P, compute_loss=False)
        P_s = out["P_simplified"]          # (B, simplifier_M, 3)
        if P_s.shape[1] == M:
            return P_s
        elif P_s.shape[1] > M:
            return fps(P_s, M)             # FPS subsample dari simplified
        else:
            return fps(P, M)               # fallback

    # ── Define all computed methods ────────────────────────────────────
    computed_methods = {
        "Ours":   our_fn,
        "FPS":    fps,
        "Random": random_sample,
        "Voxel":  voxel_downsample,
    }

    # ── Sweep M × Method ──────────────────────────────────────────────
    run_results: dict[str, dict[int, dict]] = {name: {} for name in computed_methods}

    for M in args.M_list:
        logger.info(f"{'='*55}")
        logger.info(f"  M = {M}")
        logger.info(f"{'='*55}")

        for name, fn in computed_methods.items():
            logger.info(f"  [{name}] evaluating...")
            r = evaluate_method(
                pointnet, loader, fn, M, device, args.num_class, label=name,
            )
            run_results[name][M] = r
            logger.info(
                f"  [{name}]  OA={r['oa']:.2f}%  "
                f"mAcc={r['macc']:.2f}%  "
                f"time={r['avg_time_ms']:.1f}ms/sample"
            )

    # ── Print ASCII table ──────────────────────────────────────────────
    print_table(args.M_list, run_results, full_oa, args.num_class)

    # ── Gap analysis vs SOTA ───────────────────────────────────────────
    print("  Gap vs SOTA (Ours − APES):")
    for M in args.M_list:
        our_oa  = run_results["Ours"].get(M, {}).get("oa")
        apes_oa = SOTA_PUBLISHED["APES"].get(M)
        if our_oa is not None and apes_oa is not None:
            gap = our_oa - apes_oa
            sym = "✓" if gap >= 0 else "✗"
            print(f"    M={M:<4}  {our_oa:.2f}% − {apes_oa:.2f}% = {gap:+.2f}%  {sym}")

    # ── Save JSON ──────────────────────────────────────────────────────
    out_json = out_dir / "pointnet_protocol_results.json"
    payload  = {
        "protocol":    "frozen_pointnet",
        "dataset":     args.dataset,
        "full_cloud":  full_res,
        "M_list":      args.M_list,
        "computed":    {k: {str(m): v for m, v in d.items()}
                        for k, d in run_results.items()},
        "sota_published": {k: {str(m): v for m, v in d.items() if v is not None}
                           for k, d in SOTA_PUBLISHED.items()},
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results JSON → {out_json}")

    # ── Figure ────────────────────────────────────────────────────────
    fig_path = out_dir / "pointnet_protocol_table.png"
    make_figure(args.M_list, run_results, full_oa, fig_path)

    logger.info(f"Done. Semua output di {out_dir}/")
    print("Paste ke paper lo:")
    print(f"  Full cloud (N=1024): {full_oa:.2f}%")
    for M in args.M_list:
        our = run_results['Ours'].get(M, {}).get('oa', '—')
        print(f"  Ours  M={M}: {our:.2f}%" if isinstance(our, float) else f"  Ours  M={M}: {our}")


if __name__ == "__main__":
    main()
