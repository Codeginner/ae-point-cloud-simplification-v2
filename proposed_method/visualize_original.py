"""
visualize_original.py
=====================
Visualisasi original point cloud 1024 titik dengan warna abu-abu,
background putih — siap embed ke diagram arsitektur.

Usage:
------
    # Mode dummy (sintetis, tanpa dataset):
    python visualize_original.py --dummy --output original_cloud.png

    # Mode dataset HDF5 (ModelNet40 Kaggle):
    python visualize_original.py \
        --data_root  /kaggle/working/ae-point-cloud-simplification-v2/data/modelnet40_ply_hdf5_2048 \
        --data_format hdf5 \
        --sample_idx 0 \
        --output original_cloud.png

    # Ganti objek (cari by label name):
    python visualize_original.py ... --class_name chair

    # Simpan banyak sudut pandang sekaligus:
    python visualize_original.py ... --multiview

Output:
-------
    original_cloud.png            — tampilan tunggal
    original_cloud_multiview.png  — 4 sudut pandang (jika --multiview)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Konfigurasi visual — sesuaikan di sini
# ---------------------------------------------------------------------------

POINT_COLOR  = "#5a5a5a"   # abu-abu gelap, mirip gambar referensi
POINT_SIZE   = 2.0         # ukuran titik (matplotlib scatter s=)
POINT_ALPHA  = 0.75        # transparansi
BG_COLOR     = "white"
FIG_SIZE     = (5, 5)      # inch
DPI          = 300
ELEV_DEFAULT = 20
AZIM_DEFAULT = -50


# ---------------------------------------------------------------------------
# Dummy airplane / chair sintetis
# ---------------------------------------------------------------------------

MN40_CLASSES = [
    "airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car",
    "chair","cone","cup","curtain","desk","door","dresser","flower_pot",
    "glass_box","guitar","keyboard","lamp","laptop","mantel","monitor",
    "night_stand","person","piano","plant","radio","range_hood","sink",
    "sofa","stairs","stool","table","tent","toilet","tv_stand","vase",
    "wardrobe","xbox",
]


def _make_dummy_chair(n: int = 1024, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []

    def slab(x0, x1, y0, y1, z0, z1, count):
        return rng.uniform(
            [x0, y0, z0], [x1, y1, z1], (count, 3)
        ).astype(np.float32)

    # dudukan
    pts.append(slab(-0.45, 0.45, 0.0, 0.06, -0.45, 0.45, 260))
    # sandaran
    pts.append(slab(-0.45, 0.45, 0.0, 0.65, 0.40, 0.48, 200))
    # 4 kaki
    for xc, zc in [(-0.38, -0.38), (0.38, -0.38), (-0.38, 0.38), (0.38, 0.38)]:
        pts.append(slab(xc-0.04, xc+0.04, -0.60, 0.0, zc-0.04, zc+0.04, 80))

    pts = np.concatenate(pts, axis=0)
    idx = rng.choice(len(pts), n, replace=len(pts) < n)
    pts = pts[idx]

    # normalise unit sphere
    pts -= pts.mean(axis=0)
    pts /= (np.linalg.norm(pts, axis=1).max() + 1e-8)
    return pts


def _make_dummy_airplane(n: int = 1024, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []

    t   = rng.uniform(0, 2*np.pi, 320)
    phi = rng.uniform(0, np.pi, 320)
    parts.append(np.stack([
        0.9*np.cos(phi),
        0.12*np.sin(phi)*np.cos(t),
        0.10*np.sin(phi)*np.sin(t),
    ], axis=1))

    for sign in (-1, 1):
        u = rng.uniform(-0.3, 0.3, 200)
        v = rng.uniform(0.0, 0.5, 200) * sign
        w = rng.uniform(-0.02, 0.02, 200)
        parts.append(np.stack([u, w, v], axis=1))

    u = rng.uniform(0.55, 0.85, 100)
    parts.append(np.stack([u, rng.uniform(0.0, 0.25, 100), rng.uniform(-0.02, 0.02, 100)], axis=1))
    for sign in (-1, 1):
        u = rng.uniform(0.55, 0.80, 102)
        v = rng.uniform(0.05, 0.20, 102) * sign
        parts.append(np.stack([u, rng.uniform(-0.02, 0.02, 102), v], axis=1))

    pts = np.concatenate(parts, axis=0).astype(np.float32)
    idx = rng.choice(len(pts), n, replace=len(pts) < n)
    pts = pts[idx]
    pts -= pts.mean(axis=0)
    pts /= (np.linalg.norm(pts, axis=1).max() + 1e-8)
    return pts


# ---------------------------------------------------------------------------
# Load dari HDF5 (ModelNet40 format APES/SampleNet)
# ---------------------------------------------------------------------------

def load_from_hdf5(data_root: str, mode: str, sample_idx: int,
                   n_points: int = 1024, class_name: str = None):
    import glob, h5py

    files = sorted(glob.glob(f"{data_root}/{mode}*.h5"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file .h5 di {data_root} dengan prefix '{mode}'")

    all_pts, all_labels = [], []
    for f in files:
        with h5py.File(f, "r") as hf:
            all_pts.append(hf["data"][:])
            all_labels.append(hf["label"][:].flatten())

    all_pts    = np.concatenate(all_pts,    axis=0)   # (N_total, 2048, 3)
    all_labels = np.concatenate(all_labels, axis=0)   # (N_total,)

    if class_name is not None:
        class_name = class_name.lower()
        if class_name not in MN40_CLASSES:
            raise ValueError(f"class '{class_name}' tidak ada. Pilih dari: {MN40_CLASSES}")
        target_label = MN40_CLASSES.index(class_name)
        mask = all_labels == target_label
        all_pts = all_pts[mask]
        print(f"[filter] class='{class_name}' (label {target_label}) → {all_pts.shape[0]} sampel")
        sample_idx = min(sample_idx, len(all_pts) - 1)

    pts = all_pts[sample_idx].astype(np.float32)   # (2048, 3)
    label = int(all_labels[sample_idx])

    # subsample
    rng = np.random.default_rng(0)
    if n_points < pts.shape[0]:
        idx = rng.choice(pts.shape[0], n_points, replace=False)
        pts = pts[idx]

    # normalise
    pts -= pts.mean(axis=0)
    pts /= (np.linalg.norm(pts, axis=1).max() + 1e-8)

    class_str = MN40_CLASSES[label] if label < len(MN40_CLASSES) else str(label)
    print(f"[data]   sample_idx={sample_idx}  label={label} ({class_str})  shape={pts.shape}")
    return pts, class_str


# ---------------------------------------------------------------------------
# Load dari NPY (format repo ini)
# ---------------------------------------------------------------------------

def load_from_npy(data_root: str, dataset: str, mode: str,
                  sample_idx: int, n_points: int = 1024):
    import glob
    pcd_dir = f"{data_root}/{dataset}/pcd/{mode}"
    files   = sorted(glob.glob(f"{pcd_dir}/*.npy"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file .npy di {pcd_dir}")

    pts = np.load(files[sample_idx]).astype(np.float32)

    rng = np.random.default_rng(0)
    if n_points < len(pts):
        idx = rng.choice(len(pts), n_points, replace=False)
        pts = pts[idx]

    pts -= pts.mean(axis=0)
    pts /= (np.linalg.norm(pts, axis=1).max() + 1e-8)
    print(f"[data]   sample_idx={sample_idx}  shape={pts.shape}")
    return pts, "object"


# ---------------------------------------------------------------------------
# Render satu sudut pandang
# ---------------------------------------------------------------------------

def render_single(
    ax,
    pts:   np.ndarray,
    elev:  float = ELEV_DEFAULT,
    azim:  float = AZIM_DEFAULT,
    title: str   = "",
    color: str   = POINT_COLOR,
    size:  float = POINT_SIZE,
    alpha: float = POINT_ALPHA,
):
    # matplotlib 3d: x=x, y=z (depth), z=y (height) → agar Y ke atas
    ax.scatter(
        pts[:, 0], pts[:, 2], pts[:, 1],
        c=color,
        s=size,
        alpha=alpha,
        linewidths=0,
        depthshade=True,
        rasterized=True,
    )
    ax.set_facecolor(BG_COLOR)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    if title:
        ax.set_title(title, fontsize=9, color="#333333", pad=4)


# ---------------------------------------------------------------------------
# Simpan satu gambar
# ---------------------------------------------------------------------------

def save_single(pts: np.ndarray, output: str,
                elev: float = ELEV_DEFAULT, azim: float = AZIM_DEFAULT,
                label: str = ""):
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG_COLOR)
    render_single(ax, pts, elev=elev, azim=azim)
    plt.tight_layout(pad=0)
    fig.savefig(output, dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR, transparent=False)
    plt.close(fig)
    print(f"  saved → {output}")


# ---------------------------------------------------------------------------
# Simpan multiview (4 sudut)
# ---------------------------------------------------------------------------

def save_multiview(pts: np.ndarray, output: str, label: str = ""):
    views = [
        (20, -50,  "view 1"),
        (20,  40,  "view 2"),
        (20, 130,  "view 3"),
        (60, -50,  "top"),
    ]
    fig = plt.figure(figsize=(FIG_SIZE[0]*4, FIG_SIZE[1]), facecolor=BG_COLOR)
    for i, (elev, azim, name) in enumerate(views):
        ax = fig.add_subplot(1, 4, i+1, projection="3d", facecolor=BG_COLOR)
        render_single(ax, pts, elev=elev, azim=azim)
    plt.tight_layout(pad=0.2)
    fig.savefig(output, dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR, transparent=False)
    plt.close(fig)
    print(f"  saved → {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualisasi original point cloud 1024 titik (abu-abu)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dummy",       action="store_true",
                   help="Gunakan point cloud sintetis (chair)")
    p.add_argument("--dummy_class", type=str, default="chair",
                   choices=["chair", "airplane"],
                   help="Kelas sintetis jika --dummy")

    p.add_argument("--data_root",   type=str, default="./data")
    p.add_argument("--dataset",     type=str, default="modelnet40")
    p.add_argument("--data_format", type=str, default="hdf5",
                   choices=["hdf5", "npy"])
    p.add_argument("--mode",        type=str, default="test")
    p.add_argument("--n_points",    type=int, default=1024)
    p.add_argument("--sample_idx",  type=int, default=0)
    p.add_argument("--class_name",  type=str, default=None,
                   help="Filter by class name, e.g. 'chair', 'airplane'")

    p.add_argument("--elev",  type=float, default=ELEV_DEFAULT)
    p.add_argument("--azim",  type=float, default=AZIM_DEFAULT)

    p.add_argument("--color", type=str, default=POINT_COLOR,
                   help="Warna titik, hex atau nama matplotlib")
    p.add_argument("--size",  type=float, default=POINT_SIZE)
    p.add_argument("--alpha", type=float, default=POINT_ALPHA)

    p.add_argument("--output",     type=str, default="original_cloud.png")
    p.add_argument("--multiview",  action="store_true",
                   help="Simpan juga 4 sudut pandang sekaligus")
    return p.parse_args()


def main():
    args = parse_args()

    global POINT_COLOR, POINT_SIZE, POINT_ALPHA
    POINT_COLOR = args.color
    POINT_SIZE  = args.size
    POINT_ALPHA = args.alpha

    print("=" * 50)
    print("  Original Point Cloud Visualizer")
    print("=" * 50)

    if args.dummy:
        label = args.dummy_class
        pts = (_make_dummy_chair() if label == "chair" else _make_dummy_airplane())
        print(f"[mode]   dummy ({label})  shape={pts.shape}")
    elif args.data_format == "hdf5":
        pts, label = load_from_hdf5(
            args.data_root, args.mode, args.sample_idx,
            n_points=args.n_points, class_name=args.class_name,
        )
    else:
        pts, label = load_from_npy(
            args.data_root, args.dataset, args.mode,
            args.sample_idx, n_points=args.n_points,
        )

    print(f"[render] elev={args.elev}  azim={args.azim}  "
          f"color={args.color}  size={args.size}  alpha={args.alpha}")

    # --- main output ---
    out_path = Path(args.output)
    save_single(pts, str(out_path), elev=args.elev, azim=args.azim, label=label)

    # --- multiview ---
    if args.multiview:
        mv_path = out_path.parent / f"{out_path.stem}_multiview.png"
        save_multiview(pts, str(mv_path), label=label)

    print("=" * 50)
    print("  selesai.")
    print("=" * 50)


if __name__ == "__main__":
    main()
