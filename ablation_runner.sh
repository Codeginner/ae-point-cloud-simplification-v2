#!/bin/bash
# ablation_runner.sh
# ==================
# Runs all ablation variants and saves results to ./eval_output/ablation/
#
# Usage:
#   bash ablation_runner.sh
#
# Assumes:
#   - Training is done and best.pth exists
#   - pointnet_cls_mn40.pth exists
#   - evaluate_all_extended.py is in the same directory

set -e  # exit on any error

DATA_ROOT="./data"
POINTNET_CKPT="./checkpoints/pointnet_cls_mn40.pth"
OUT_DIR="./eval_output/ablation"
M=512   # ablation only needs M=512

echo "========================================"
echo " ABLATION STUDY RUNNER"
echo "========================================"

# ── Helper: train + evaluate one variant ──────────────────────────────────────
run_variant() {
    local TAG=$1
    local EXTRA_ARGS=$2
    local CKPT="./checkpoints/ablation_${TAG}.pth"

    echo ""
    echo "----------------------------------------"
    echo "[VARIANT] ${TAG}"
    echo "----------------------------------------"

    # Train
    torchrun --nproc_per_node=2 proposed_method/train_ddp.py \
        --dataset    modelnet40 \
        --data_root  ${DATA_ROOT}/modelnet40_ply_hdf5_2048 \
        --data_format hdf5 \
        --M          ${M} \
        --epochs     200 \
        --batch_size 32 \
        --lr         1e-4 \
        --checkpoint_out ${CKPT} \
        ${EXTRA_ARGS}

    # Evaluate
    python evaluate_all_extended.py \
        --checkpoint    ${CKPT} \
        --pointnet_ckpt ${POINTNET_CKPT} \
        --data_root     ${DATA_ROOT} \
        --M_list        ${M} \
        --variant       "${TAG}" \
        --out           ${OUT_DIR}/${TAG}

    echo "[DONE] ${TAG}"
}

# ── 1. Full proposed (already trained — just re-evaluate for ablation table) ──
echo ""
echo "[STEP 1/4] Evaluating full proposed model..."
python evaluate_all_extended.py \
    --checkpoint    ./checkpoints/best.pth \
    --pointnet_ckpt ${POINTNET_CKPT} \
    --data_root     ${DATA_ROOT} \
    --M_list        ${M} \
    --variant       "Proposed (full)" \
    --out           ${OUT_DIR}/full

# ── 2. w/o geometry balance: alpha=1.0  (all points to contour, no planar) ───
run_variant "no_balance_alpha1" \
    "--alpha 1.0 --lambda_cls 1.0 --lambda_4 0.1"

# ── 3. w/o contour awareness: alpha=0.0  (all points to planar) ──────────────
run_variant "no_contour_alpha0" \
    "--alpha 0.0 --lambda_cls 1.0 --lambda_4 0.1"

# ── 4. w/o lambda_4 term ─────────────────────────────────────────────────────
run_variant "no_lambda4" \
    "--alpha 0.6 --lambda_cls 1.0 --lambda_4 0.0"

# ── Merge all ablation JSONs into one summary ─────────────────────────────────
echo ""
echo "========================================"
echo " Merging ablation results..."
echo "========================================"

python - <<'EOF'
import json, os, glob
from pathlib import Path

rows = {}
for path in sorted(glob.glob('./eval_output/ablation/**/eval_results.json', recursive=True)):
    with open(path) as f:
        data = json.load(f)
    # data is {variant_name: {M: {oa, cd, time_ms}}}
    for variant, m_dict in data.items():
        for M, metrics in m_dict.items():
            rows[variant] = metrics

out = Path('./eval_output/ablation_summary.json')
with open(out, 'w') as f:
    json.dump(rows, f, indent=2)

print("\nAblation Summary")
print("="*55)
print(f"{'Variant':<30}  {'OA (%)':>7}  {'CD':>10}")
print("-"*55)
for variant, metrics in rows.items():
    print(f"{variant:<30}  {metrics.get('oa','N/A'):>7}  {metrics.get('cd','N/A'):>10}")

print(f"\nSaved to {out}")
EOF

echo ""
echo "========================================"
echo " ALL ABLATION VARIANTS COMPLETE"
echo "========================================"
