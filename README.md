```bash
# Single GPU
python run_train.py \
    --dataset modelnet40 \
    --M 512 \
    --epochs 250 \
    --batch_size 32 \
    --lr 1e-3
 
# DDP 2 gpu
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
