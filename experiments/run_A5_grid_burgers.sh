#!/bin/bash
# REVISION A5 网格: Burgers 全 lr×R cell 的 Γᵃ=0 (a_ks=[1,1,1],a_perp=1.0) vs full STRATA
# 验证 Cor 18.3 "Γᵃ=0 旗几何失效" 在 off-optimum 也成立(不只单个最优 cell)
# full STRATA 已有 runs2_grid_eps1e6/burgers__strata__*; 这里只跑 Γᵃ=0 对照
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs_revision_A5 logs_revision
# Activate the project Python environment before running this launcher.
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU
for lr in 5e-3 1e-3 5e-4; do for R in 32 64; do
  out=runs_revision_A5/burgers__A5gamma0__lr${lr}__R${R}
  [ -f $out/summary.json ] && { echo "skip $out"; continue; }
  python -u train_pinn.py --method strata --epochs 3000 --lr $lr --R $R --seed 42 \
    --warmup 100 --hidden 128 --depth 5 --n_col 20000 --xi 1e-3 --tau 50 --p 8 \
    --a_ks 1.0 1.0 1.0 --a_perp 1.0 --flag_log_every 100 \
    --out $out > logs_revision/A5_burgers_${lr}_${R}.log 2>&1
  echo "done $out"
done; done
echo "ALL A5 BURGERS GRID DONE"
