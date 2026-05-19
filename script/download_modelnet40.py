#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_modelnet10.py

Download ModelNet40 (HDF5 format) dan filter jadi 10 kelas subset:
    bathtub, bed, chair, desk, dresser,
    monitor, night_stand, sofa, table, toilet

Usage:
    python download_modelnet10.py
    python download_modelnet10.py --data_root ./data
"""

import os
import glob
import argparse
import h5py
import numpy as np


# Label asli ModelNet40 (alphabetical order, index 0-39)
MODELNET40_CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
    'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser',
    'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
    'laptop', 'mantel', 'monitor', 'night_stand', 'person',
    'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent',
    'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(url: str, data_root: str) -> str:
    """Download dan unzip ModelNet40 HDF5. Return path folder hasil unzip."""
    zipfile   = os.path.basename(url)
    folder    = zipfile.replace('.zip', '')
    zip_path  = os.path.join(data_root, zipfile)
    dest_path = os.path.join(data_root, folder)

    if os.path.exists(dest_path):
        print(f"[skip] {dest_path} sudah ada, skip download.")
        return dest_path

    os.makedirs(data_root, exist_ok=True)
    print(f"Downloading {url} ...")
    os.system(f'wget "{url}" --no-check-certificate -O "{zip_path}"')

    print(f"Extracting {zip_path} ...")
    os.system(f'unzip -q "{zip_path}" -d "{data_root}"')
    os.system(f'rm "{zip_path}"')

    return dest_path


# ---------------------------------------------------------------------------
# Read HDF5
# ---------------------------------------------------------------------------

def read_data(files: list) -> tuple:
    """Baca semua HDF5 dan gabungkan.

    Returns:
        all_pcd:   (N_total, 2048, 3)  float32
        all_label: (N_total,)          uint8
    """
    all_pcd, all_label = [], []
    for h5_name in sorted(files):
        print(f"  reading {os.path.basename(h5_name)} ...")
        with h5py.File(h5_name, 'r') as f:
            pcd   = f['data'][:].astype('float32')    # (N, 2048, 3)
            label = f['label'][:].astype('uint8')      # (N, 1)
        all_pcd.append(pcd)
        all_label.append(label[:, 0])

    all_pcd   = np.concatenate(all_pcd,   axis=0)     # (N_total, 2048, 3)
    all_label = np.concatenate(all_label, axis=0)     # (N_total,)
    return all_pcd, all_label



# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_data(data_root: str, pcds: np.ndarray, labels: np.ndarray, mode: str) -> None:
    """Simpan tiap point cloud dan label sebagai file .npy terpisah."""
    pcd_dir   = os.path.join(data_root, 'modelnet40', 'pcd',   mode)
    label_dir = os.path.join(data_root, 'modelnet40', 'label', mode)
    os.makedirs(pcd_dir,   exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    print(f"Saving {mode} data ({len(pcds)} samples) ...")
    for i, (pcd, label) in enumerate(zip(pcds, labels)):
        np.save(os.path.join(pcd_dir,   f'{i:04d}'), pcd)    # (2048, 3)
        np.save(os.path.join(label_dir, f'{i:04d}'), label)  # scalar
        if (i + 1) % 500 == 0 or (i + 1) == len(pcds):
            print(f"  {i + 1}/{len(pcds)}")

    print(f"Done saving {mode}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(url: str, data_root: str) -> None:
    out_dir = os.path.join(data_root, 'modelnet40')
    if os.path.exists(out_dir):
        print(f"[skip] {out_dir} sudah ada. Hapus folder ini kalau mau download ulang.")
        return

    # 1. Download
    h5_folder = download(url, data_root)

    # 2. Train
    print("\n--- TRAIN ---")
    train_files = glob.glob(os.path.join(h5_folder, '*train*.h5'))
    train_pcds, train_labels = read_data(train_files)
    print(f"total train samples: {len(train_labels)}")
    save_data(data_root, train_pcds, train_labels, 'train')

    # 3. Test
    print("\n--- TEST ---")
    test_files = glob.glob(os.path.join(h5_folder, '*test*.h5'))
    test_pcds, test_labels = read_data(test_files)
    print(f"total test samples: {len(test_labels)}")
    save_data(data_root, test_pcds, test_labels, 'test')

    # 4. Simpan class list buat referensi
    class_file = os.path.join(out_dir, 'classes.txt')
    with open(class_file, 'w') as f:
        for i, cls in enumerate(MODELNET40_CLASSES):
            f.write(f'{i}\t{cls}\n')
    print(f"\nClass list saved to {class_file}")

    # 5. Cleanup HDF5
    print(f"\nCleaning up {h5_folder} ...")
    os.system(f'rm -rf "{h5_folder}"')

    print("\nSelesai! Struktur output:")
    print(f"  {data_root}/modelnet10/")
    print(f"    pcd/train/0000.npy ... (2048, 3)")
    print(f"    pcd/test/0000.npy")
    print(f"    label/train/0000.npy  (scalar 0-9)")
    print(f"    label/test/0000.npy")
    print(f"    classes.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--url', type=str,
                        default='https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip')
    args = parser.parse_args()

    main(args.url, args.data_root)