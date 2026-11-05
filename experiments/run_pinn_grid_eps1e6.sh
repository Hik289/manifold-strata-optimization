#!/bin/bash
# PINN 对称网格重跑 (eps=1e-6 已是默认). H3 重判关键。
# STRATA + Single-Grassmann × {burgers, burgersg2, poisson} × lr{5e-3,1e-3,5e-4}×R{32,64}
# 同之前公平协议, 新目录 runs2_grid_eps1e6/
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs2_grid_eps1e6 logs
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU
LRS="5e-3 1e-3 5e-4"
RS="32 64"

run_one () {
  local script=$1 method=$2 tag=$3 extra=$4
  for lr in $LRS; do for R in $RS; do
    local out=runs2_grid_eps1e6/${tag}__${method}__lr${lr}__R${R}
    [ -f $out/summary.json ] && { echo "skip $out"; continue; }
    if [ "$method" = "strata" ]; then AKS="--a_ks 1.0 0.7 0.4 --a_perp 0.2 --xi 1e-3 --tau 50 --p 8";
    else AKS="--a_perp 0.2 --xi 1e-3 --tau 50 --p 8"; fi
    python -u $script --method $method --epochs 3000 --lr $lr --R $R --seed 42 \
      --warmup 100 --flag_log_every 100 $extra $AKS --out $out > logs/eps6_${tag}_${method}_${lr}_${R}.log 2>&1
    echo "done $out"
  done; done
}

run_one train_pinn.py strata           burgers   "--hidden 128 --depth 5 --n_col 20000"
run_one train_pinn.py single_grassmann burgers   "--hidden 128 --depth 5 --n_col 20000"
run_one train_pinn.py strata           burgersg2 "--hidden 96 --depth 4 --n_col 20000"
run_one train_pinn.py single_grassmann burgersg2 "--hidden 96 --depth 4 --n_col 20000"
run_one train_poisson.py strata           poisson "--hidden 128 --depth 5 --n_col 5000"
run_one train_poisson.py single_grassmann poisson "--hidden 128 --depth 5 --n_col 5000"
echo "ALL PINN eps1e6 GRID DONE"
