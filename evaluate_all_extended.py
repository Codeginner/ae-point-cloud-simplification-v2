"""
evaluate_all_extended.py
========================
Extended evaluation script for point cloud simplification.

Covers:
  - Methods : RS, FPS, APES, Proposed
  - Metrics : OA (downstream PointNet), CD (Chamfer Distance)
  - Ratios  : M in [512, 256, 128, 64, 32]
  - Extras  : per-method runtime, ablation support via --variant flag

Usage
-----
# Full evaluation (all methods, all ratios)
python evaluate_all_extended.py \
    --checkpoint    ./checkpoints/best.pth \
    --pointnet_ckpt ./checkpoints/pointnet_cls_mn40.pth \
    --data_root     ./data \
    --M_list 512 256 128 64 32 \
    --dataset       modelnet40

# Ablation variant (pass variant name for logging)
python evaluate_all_extended.py \
    --checkpoint    ./checkpoints/ablation_no_lambda4.pth \
    --pointnet_ckpt ./checkpoints/pointnet_cls_mn40.pth \
    --data_root     ./data \
    --M_list 512 \
    --dataset       modelnet40 \
    --variant       "w/o lambda4"
"""

import os
import time
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── adjust these imports to match your project layout ──────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "script"))

from proposed_method.model import ProposedSimplifier          # your model
from proposed_method.dataset import ModelNet40Dataset         # your dataloader
# from baselines.apes import APESSampler                      # APES skipped — refer to paper
from script.train_pointnet import PointNetCls                 # pretrained PointNet
# ───────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def random_sampling(pc: torch.Tensor, M: int) -> torch.Tensor:
    """pc: (B, N, 3)  →  (B, M, 3)"""
    B, N, _ = pc.shape
    idx = torch.stack([torch.randperm(N)[:M] for _ in range(B)], dim=0).to(pc.device)
    return pc[torch.arange(B).unsqueeze(1), idx]


def fps(pc: torch.Tensor, M: int) -> torch.Tensor:
    """Farthest Point Sampling. pc: (B, N, 3)  →  (B, M, 3)"""
    B, N, _ = pc.shape
    device = pc.device
    selected = torch.zeros(B, M, dtype=torch.long, device=device)
    dist = torch.full((B, N), float('inf'), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(M):
        selected[:, i] = farthest
        centroid = pc[torch.arange(B), farthest].unsqueeze(1)   # (B,1,3)
        d = ((pc - centroid) ** 2).sum(-1)                       # (B,N)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)

    return pc[torch.arange(B).unsqueeze(1), selected]


# ─────────────────────────────────────────────────────────────────────────────
#  Chamfer Distance  (symmetric, mean)
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> float:
    """
    p1, p2: (B, M, 3) and (B, N, 3)
    Returns mean symmetric CD over the batch.
    """
    p1 = p1.unsqueeze(2)   # (B, M, 1, 3)
    p2 = p2.unsqueeze(1)   # (B, 1, N, 3)
    dist = ((p1 - p2) ** 2).sum(-1)            # (B, M, N)
    cd = dist.min(2)[0].mean() + dist.min(1)[0].mean()
    return cd.item()


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = ModelNet40Dataset(root=args.data_root, split='test', npoints=1024)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)

    # ── load models ───────────────────────────────────────────────────────────
    # FIXED
    '''
    proposed = ProposedSimplifier(num_class=40).to(device)
    ckpt_proposed = torch.load(args.checkpoint, map_location=device)
    proposed.load_state_dict(ckpt_proposed['model_state_dict'] if 'model_state_dict' in ckpt_proposed else ckpt_proposed)
    proposed.eval()
    '''
    ckpt_proposed = torch.load(args.checkpoint, map_location=device)
    ckpt_sd = ckpt_proposed['model_state_dict'] if 'model_state_dict' in ckpt_proposed else ckpt_proposed

    def load_proposed(M):
        m = ProposedSimplifier(num_class=40, M=M).to(device)
        m.load_state_dict(ckpt_sd)
        m.eval()
        return m
    
    pointnet = PointNetCls().to(device)
    ckpt_pn = torch.load(args.pointnet_ckpt, map_location=device)
    pointnet.load_state_dict(ckpt_pn['model'] if 'model' in ckpt_pn else ckpt_pn)
    pointnet.eval()

    # APES skipped — results taken from paper
    # apes = APESSampler().to(device)

    # ── methods to evaluate ───────────────────────────────────────────────────
    #    Each entry: (display_name, callable(pc, M) → simplified_pc)
    methods = {
        'Random Sampling': lambda pc, M: random_sampling(pc, M),
        'FPS':             lambda pc, M: fps(pc, M),
        # 'APES':          lambda pc, M: apes(pc, M),   # refer to paper
        'Proposed':        lambda pc, M: load_proposed(M)(pc, labels=None, compute_loss=False)['P_simplified'],
    }

    results = {}   # results[method][M] = {'oa': float, 'cd': float, 'time_ms': float}

    for method_name, method_fn in methods.items():
        results[method_name] = {}
        for M in args.M_list:
            print(f"\n[EVAL] {method_name:20s}  M={M}")
            all_correct, all_total = 0, 0
            all_cd, all_time = [], []

            with torch.no_grad():
                for batch in loader:
                    # ── unpack ─────────────────────────────────────────────
                    if isinstance(batch, (list, tuple)):
                        pc, labels = batch[0].to(device), batch[1].to(device)
                    else:
                        pc     = batch['points'].to(device)
                        labels = batch['label'].to(device)

                    # ── sample + time ──────────────────────────────────────
                    t0 = time.perf_counter()
                    simplified = method_fn(pc, M)
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    # ── CD ─────────────────────────────────────────────────
                    cd = chamfer_distance(simplified, pc)
                    all_cd.append(cd)
                    all_time.append(elapsed_ms / pc.shape[0])   # per-sample

                    # ── OA  (upsample back to 1024 for PointNet if needed) ─
                    # PointNet is permutation-invariant; pad with zeros if M < 1024
                    if simplified.shape[1] < 1024:
                        pad = torch.zeros(
                            simplified.shape[0], 1024 - simplified.shape[1], 3,
                            device=device
                        )
                        input_pc = torch.cat([simplified, pad], dim=1)
                    else:
                        input_pc = simplified

                    logits = pointnet(input_pc)
                    preds  = logits.argmax(-1)
                    all_correct += (preds == labels).sum().item()
                    all_total   += labels.shape[0]

            oa    = 100.0 * all_correct / all_total
            mean_cd   = float(np.mean(all_cd))
            mean_time = float(np.mean(all_time))

            results[method_name][M] = {
                'oa':      round(oa, 2),
                'cd':      round(mean_cd, 6),
                'time_ms': round(mean_time, 3),
            }
            print(f"  OA={oa:.2f}%  CD={mean_cd:.6f}  Time={mean_time:.2f}ms/sample")

    # ── attach variant label if this is an ablation run ───────────────────────
    if args.variant:
        results = {args.variant: results.get('Proposed', {})}

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.out) / 'eval_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {out_path}")

    # ── pretty-print tables ───────────────────────────────────────────────────
    print_tables(results, args.M_list)
    return results


def print_tables(results, M_list):
    methods = list(results.keys())

    # OA table
    print("\n" + "="*70)
    print("OVERALL ACCURACY (%)")
    print("="*70)
    header = f"{'Method':<22}" + "".join(f"  M={M:<5}" for M in M_list)
    print(header)
    print("-"*70)
    for m in methods:
        row = f"{m:<22}"
        for M in M_list:
            val = results[m].get(M, {}).get('oa', 'N/A')
            row += f"  {val:<7}"
        print(row)

    # CD table
    print("\n" + "="*70)
    print("CHAMFER DISTANCE (lower is better)")
    print("="*70)
    print(header)
    print("-"*70)
    for m in methods:
        row = f"{m:<22}"
        for M in M_list:
            val = results[m].get(M, {}).get('cd', 'N/A')
            row += f"  {val:<7}"
        print(row)

    # Runtime table
    print("\n" + "="*70)
    print("RUNTIME (ms / sample)")
    print("="*70)
    print(header)
    print("-"*70)
    for m in methods:
        row = f"{m:<22}"
        for M in M_list:
            val = results[m].get(M, {}).get('time_ms', 'N/A')
            row += f"  {val:<7}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--pointnet_ckpt', required=True)
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--M_list',        nargs='+', type=int,
                        default=[512, 256, 128, 64, 32])
    parser.add_argument('--dataset',       default='modelnet40')
    parser.add_argument('--variant',       default='',
                        help='Label for ablation run (optional)')
    parser.add_argument('--out',           default='./eval_output')
    args = parser.parse_args()

    evaluate(args)
