#!/bin/bash
# RUNNING-wave1 (修正版): CIFAR-100 / CompactViT, 1000 epochs, seed=42
# 修复: bias-correction counter 分离 (c_M 不重置/c_V 重置, §3.8 算法修正, implementation 已认可)
# 加固: §5.3 hook 8 显式 principal-angle 漂移 + level_energy None 兜底+collapse flag
# 新目录 runs2/ (保留原始发散 run 在 runs/)
# 5 methods, matched max rank R=64, packed across 3 GPUs (compute-bound, ~2GB mem each)
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs2 logs

EP=1000
LR=3e-4
WD=0.05
R=64
COMMON="--epochs $EP --lr $LR --wd $WD --R $R --seed 42 --warmup 20 --flag_log_every 10"

# GPU2: STRATA (最重, 单独一卡). 非平凡 a_ks 保证 Γᵃ>0 (Corollary 18.3)
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --method strata $COMMON \
  --xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2 \
  --out runs2/strata > logs/strata2.log 2>&1 &
echo "GPU2 strata pid $!"

# GPU1: Single-Grassmann (H3 主对照, 同 R 同 a_perp) + GaLore
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --method single_grassmann $COMMON \
  --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 \
  --out runs2/single_grassmann > logs/single_grassmann2.log 2>&1 &
echo "GPU1 single_grassmann pid $!"

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --method galore $COMMON \
  --galore_tau 200 --galore_scale 0.25 \
  --out runs2/galore > logs/galore2.log 2>&1 &
echo "GPU1 galore pid $!"

# GPU0: full AdamW (性能上界) + Independent-Grassmann (A2 baseline)
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --method adamw $COMMON \
  --out runs2/adamw > logs/adamw2.log 2>&1 &
echo "GPU0 adamw pid $!"

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --method independent_grassmann $COMMON \
  --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 \
  --out runs2/independent_grassmann > logs/independent_grassmann2.log 2>&1 &
echo "GPU0 independent_grassmann pid $!"

echo "all 5 launched (runs2/)"
