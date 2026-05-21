# Fixes untuk Naikkan OA dari 87% ke ~90%+

## Root Cause Analysis

### 1. Encoder Terlalu Lemah (PALING KRITIS)
**Problem:** EdgeConv4 (256-dim) dan `conv5` fusion MLP di-comment out.
- Encoder lo hanya punya 3 EdgeConv layers tanpa fusion MLP
- Output: raw concat 64+128+256 = 448-dim (noisy, tidak di-fuse)
- APES dan DGCNN original pakai 4 EdgeConv + conv5 fusion → 1024-dim features

**Fix (`encoder.py`):**
- Restore EdgeConv4: 128×2 → 256
- Restore conv5: concat(64+64+128+256)=512 → 1024
- Output dim: 1024 (bukan 448)

### 2. ClsHead Suboptimal
**Problem:** Self-attention + std pooling terlalu complex untuk simplified points.
Self-attention bagus untuk language/sequence, tapi untuk point clouds yang irregular
(simplified points clustering di edges), EdgeConv lebih cocok karena preserve local geometry.

**Fix (`model.py`):**
- Ganti self-attention → DGCNN-style EdgeConv head
- Pooling: max(ec1) + max(ec2) → concat → MLP
- Lebih sedikit parameter, lebih cocok untuk point set structure

### 3. lambda_cls Terlalu Kecil
**Problem:** lambda_cls=0.5 artinya classification loss hanya setengah reconstruction loss.
Model lebih optimize ke reconstruction, bukan ke memilih points yang bagus untuk klasifikasi.

**Fix (`model.py`):**
- lambda_cls: 0.5 → **1.0**
- Ini yang paling langsung impact ke OA

### 4. lambda_4 (Score Supervision) Terlalu Besar
**Problem:** lambda_4=0.3 memaksa scorer output sesuai "distance-to-simplified" target.
Ini supervision yang indirect dan bisa conflict dengan cls gradient.
STE sudah handle gradient ke scorer — score supervision jadi noise.

**Fix (`model.py`):**
- lambda_4: 0.3 → **0.1**

### 5. Alpha Selector Terlalu Tinggi
**Problem:** alpha=0.7 berarti 70% points dari contour region.
Untuk klasifikasi global shape (bukan part segmentation), flat regions juga penting.
Meja datar, atap datar, dll — perlu flat points untuk distinguish class.

**Fix (`model.py`):**
- alpha: 0.7 → **0.6**

## Expected Impact

| Fix | Expected OA Gain |
|-----|-----------------|
| Restore full encoder (4 EdgeConv + conv5) | +1.5% ~ +2.5% |
| lambda_cls 0.5→1.0 | +0.5% ~ +1.0% |
| EdgeConv ClsHead | +0.3% ~ +0.5% |
| alpha 0.7→0.6 | +0.2% ~ +0.4% |
| lambda_4 0.3→0.1 | +0.1% ~ +0.2% |
| **Total** | **+2.6% ~ +4.6%** |

Target: 87% + 3.5% ≈ **90.5%** (setara APES)

## Files Yang Diubah

1. `encoder.py` — Full DGCNN restore, out_dim=1024
2. `model.py` — ClsHead, lambda_cls, lambda_4, alpha, semua in_dim updated

## Tidak Perlu Ubah

- `scoring.py` — sudah accept `in_dim` as argument, PointCloudSimplifier sekarang pass 1025
- `decoder.py` — sudah accept `in_dim` as argument, PointCloudSimplifier sekarang pass 1024
- `selector.py` — tidak ada dependency ke feature dim
- `loss.py` — tidak ada dependency ke feature dim
- `train.py` — tidak ada dependency ke feature dim

## Training Command

```bash
# Single GPU
python run_train.py \
    --dataset modelnet40 \
    --M 512 \
    --epochs 250 \
    --batch_size 32 \
    --lr 1e-3

# DDP (recommended, lebih cepat)
torchrun --nproc_per_node=2 proposed_method/train_ddp.py \
    --dataset modelnet40 \
    --M 512 \
    --epochs 250 \
    --batch_size 32 \
    --lr 1e-3 \
    --lambda_cls 1.0 \
    --lambda_4 0.1 \
    --alpha 0.6
```

## Kalau masih kurang, coba tambahan:

1. **Task-aware training dengan frozen PointNet:**
   ```bash
   torchrun ... --pointnet_ckpt ./checkpoints/pointnet_cls_mn40.pth --lambda_task 1.0
   ```
   Ini cara paling langsung — simplifier dapat gradient langsung dari PointNet.

2. **Naikkan epochs ke 300** — model dengan encoder lebih besar butuh lebih lama converge.

3. **Label smoothing di CrossEntropyLoss** — tambahkan `label_smoothing=0.1` di `F.cross_entropy`.

4. **Warmup scheduler** — seperti di `train_ddp.py` (linear warmup 5 epoch lalu cosine).
