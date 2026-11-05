#!/bin/bash
# RUNNING-wave2 SciML: Burgers PINN, 2000 epochs (SciML 硬性约束), seed=42
# 设计动机最切题的族 (高频-低频天然多时间尺度). 4 方法 + 全轨迹 δ_nest.
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs2_pinn logs
# tuned: lr=5e-3/depth5/20k col/3000ep 使 AdamW 达 rel_l2~0.029 (标准 Burgers PINN 收敛),
# 原 lr1e-3/depth4/10k/2000 是欠调 instrument (全方法卡 0.2-0.55, 非 STRATA 信号)
GPU=${1:-2}
EP=3000
COMMON="--epochs $EP --lr 5e-3 --wd 0.0 --hidden 128 --depth 5 --n_col 20000 --R 32 --seed 42 --warmup 100 --flag_log_every 30"
echo "using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU python -u train_pinn.py --method adamw $COMMON --out runs2_pinn/adamw > logs/pinn_adamw.log 2>&1
echo "adamw done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_pinn.py --method galore $COMMON --galore_tau 200 --galore_scale 0.25 --out runs2_pinn/galore > logs/pinn_galore.log 2>&1
echo "galore done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_pinn.py --method single_grassmann $COMMON --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 --out runs2_pinn/single_grassmann > logs/pinn_sg.log 2>&1
echo "single_grassmann done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_pinn.py --method strata $COMMON --xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2 --out runs2_pinn/strata > logs/pinn_strata.log 2>&1
echo "strata done"
echo "ALL 4 PINN RUNS DONE"
