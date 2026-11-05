#!/bin/bash
# wave2 PINN 机制验证: (1b) A5 消融 Γᵃ=0 + (2) baseline 公平性调参
# tuned instrument: lr5e-3/depth5/20k col/3000ep (AdamW 达 0.029)
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs2_pinn_probe logs
GPU=${1:-2}
BASE="--epochs 3000 --wd 0.0 --hidden 128 --depth 5 --n_col 20000 --R 32 --seed 42 --warmup 100 --flag_log_every 30"
echo "using GPU $GPU"

# ===== (1b) A5 消融: STRATA 所有 a_k=a_perp 相同 → Γᵃ=0 (Cor 18.3: 旗几何从更新消失) =====
# 对照: full STRATA a_ks=[1,0.7,0.4] a_perp=0.2 (Γᵃ=0.8) 已在 runs2_pinn/strata
CUDA_VISIBLE_DEVICES=$GPU python -u train_pinn.py --method strata $BASE --lr 5e-3 \
  --xi 1e-3 --tau 50 --p 8 --a_ks 1.0 1.0 1.0 --a_perp 1.0 \
  --out runs2_pinn_probe/A5_gamma0 > logs/pinn_A5.log 2>&1
echo "A5 (Γᵃ=0) done"

# ===== (2) Single-Grassmann 公平性: lr 网格 (它可能只是 lr 太大发散) =====
for LR in 1e-3 5e-4 1e-4; do
  CUDA_VISIBLE_DEVICES=$GPU python -u train_pinn.py --method single_grassmann $BASE --lr $LR \
    --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 \
    --out runs2_pinn_probe/sg_lr${LR} > logs/pinn_sg_${LR}.log 2>&1
  echo "single_grassmann lr=$LR done"
done

# ===== (2) GaLore 公平性: lr 网格 =====
for LR in 5e-3 1e-3 5e-4; do
  CUDA_VISIBLE_DEVICES=$GPU python -u train_pinn.py --method galore $BASE --lr $LR \
    --galore_tau 200 --galore_scale 0.25 \
    --out runs2_pinn_probe/galore_lr${LR} > logs/pinn_galore_${LR}.log 2>&1
  echo "galore lr=$LR done"
done
echo "ALL PROBE RUNS DONE"
