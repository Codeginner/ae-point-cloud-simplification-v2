"""
visualize_results.py
====================
Run inference on test samples, save:
  1. A multi-panel PNG (matplotlib)
  2. A JSON file compatible with the interactive 3D viewer

Usage:
    python proposed_method/visualize_results.py \
        --resume ./checkpoints/best.pth \
        --data_root ./data \
        --n_samples 8 \
        --M 512 \
        --save_dir ./viz_output
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401

# ── Make sure proposed_method is importable ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proposed_method.model   import PointCloudSimplifier
from proposed_method.train   import PointCloudDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, M: int, k: int, device) -> PointCloudSimplifier:
    model = PointCloudSimplifier(M=M, k=k).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[viz] Loaded checkpoint: {ckpt_path}")
    return model


def infer(model, P: torch.Tensor, device):
    P = P.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(P, compute_loss=False)
    return (
        P[0].cpu().numpy(),
        out["P_simplified"][0].cpu().numpy(),
        out["P_recon"][0].cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Matplotlib multi-panel figure
# ---------------------------------------------------------------------------

COLORS = {
    "original":      "#4A90D9",
    "simplified":    "#E05C5C",
    "reconstructed": "#4BC99B",
}


def plot_cloud(ax, pts, color, title, point_size=1.5):
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],   # y-up convention
               s=point_size, c=color, alpha=0.85, linewidths=0)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_axis_off()
    # Equal aspect ratio trick
    max_range = np.ptp(pts, axis=0).max() / 2
    mid       = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[2] - max_range, mid[2] + max_range)
    ax.set_zlim(mid[1] - max_range, mid[1] + max_range)


def save_figure(samples, save_path: Path):
    n   = len(samples)
    fig = plt.figure(figsize=(15, 5 * n), facecolor="#0f0f14")

    for row, (orig, simp, recon) in enumerate(samples):
        for col, (pts, color, title) in enumerate([
            (orig,  COLORS["original"],      "Original"),
            (simp,  COLORS["simplified"],    "Simplified"),
            (recon, COLORS["reconstructed"], "Reconstructed"),
        ]):
            ax = fig.add_subplot(n, 3, row * 3 + col + 1, projection="3d",
                                 facecolor="#0f0f14")
            plot_cloud(ax, pts, color, title if row == 0 else "")

    fig.text(0.5, 0.98, "Point Cloud Simplification — Results",
             ha="center", va="top", fontsize=14, color="white",
             fontweight="bold")

    plt.subplots_adjust(wspace=0.02, hspace=0.04)
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved figure → {save_path}")


# ---------------------------------------------------------------------------
# JSON export  (for interactive 3-D viewer)
# ---------------------------------------------------------------------------

def save_json(samples, save_path: Path, labels=None):
    payload = []
    for i, (orig, simp, recon) in enumerate(samples):
        payload.append({
            "id":           i,
            "label":        labels[i] if labels else f"Sample {i}",
            "original":     orig.tolist(),
            "simplified":   simp.tolist(),
            "reconstructed": recon.tolist(),
        })
    with open(save_path, "w") as f:
        json.dump(payload, f)
    print(f"[viz] Saved JSON  → {save_path}")
    print(f"      Drop this file into the interactive viewer to explore in 3-D.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume",    required=True)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--M",         type=int, default=512)
    p.add_argument("--k",         type=int, default=20)
    p.add_argument("--n_points",  type=int, default=1024)
    p.add_argument("--save_dir",  default="./viz_output")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz] device={device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model   = load_model(args.resume, args.M, args.k, device)
    dataset = PointCloudDataset(data_root=args.data_root, mode="test",
                                n_points=args.n_points, augment=False)

    n       = min(args.n_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    samples = []
    labels  = []
    for idx in indices:
        P, label = dataset[idx]
        orig, simp, recon = infer(model, P, device)
        samples.append((orig, simp, recon))
        labels.append(str(int(label)))
        print(f"  sample {idx:4d}  label={label}  "
              f"orig={orig.shape}  simp={simp.shape}  recon={recon.shape}")

    save_figure(samples, save_dir / "results.png")
    save_json(samples,   save_dir / "results.json", labels)

    print(f"\n[viz] Done — files in {save_dir}/")
    print("      Open results.png for a quick overview.")
    print("      Drag results.json into the interactive viewer for 3-D exploration.")


if __name__ == "__main__":
    main()
