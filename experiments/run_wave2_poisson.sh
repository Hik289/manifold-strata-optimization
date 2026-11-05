#!/bin/bash
# G2 第二任务: 多频 Poisson PINN (与 Burgers 不同方程, 三频天然多尺度), 3000ep
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs2_poisson logs
GPU=${1:-2}
COMMON="--epochs 3000 --lr 5e-3 --wd 0.0 --hidden 128 --depth 5 --n_col 5000 --R 32 --seed 42 --warmup 100 --flag_log_every 30"
echo "using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU python -u train_poisson.py --method adamw $COMMON --out runs2_poisson/adamw > logs/pois_adamw.log 2>&1
echo "adamw done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_poisson.py --method single_grassmann $COMMON --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 --out runs2_poisson/single_grassmann > logs/pois_sg.log 2>&1
echo "single_grassmann done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_poisson.py --method strata $COMMON --xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2 --out runs2_poisson/strata > logs/pois_strata.log 2>&1
echo "strata done"
echo "ALL POISSON RUNS DONE"
