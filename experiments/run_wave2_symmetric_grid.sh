#!/bin/bash
# 对称公平网格 (implementation 要求): STRATA 与 Single-Grassmann 用【完全相同】的 lr×R 网格,
# 相同选择协议 (val=末端100 loss_pde 无参考解泄漏, 再报 rel_l2), seed=42。
# 三任务: Burgers / Burgers_g2 / Poisson。每方法每任务 3lr×2R=6 configs。
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs2_grid logs
GPU=${1:-2}
export CUDA_VISIBLE_DEVICES=$GPU
LRS="5e-3 1e-3 5e-4"
RS="32 64"

run_one () {  # $1=script $2=method $3=task_tag $4=extra_args
  local script=$1 method=$2 tag=$3 extra=$4
  for lr in $LRS; do for R in $RS; do
    local out=runs2_grid/${tag}__${method}__lr${lr}__R${R}
    [ -f $out/summary.json ] && { echo "skip $out"; continue; }
    if [ "$method" = "strata" ]; then
      AKS="--a_ks 1.0 0.7 0.4 --a_perp 0.2 --xi 1e-3 --tau 50 --p 8"
    else
      AKS="--a_perp 0.2 --xi 1e-3 --tau 50 --p 8"
    fi
    python -u $script --method $method --epochs 3000 --lr $lr --R $R --seed 42 \
      --warmup 100 --flag_log_every 100 $extra $AKS --out $out > logs/grid_${tag}_${method}_${lr}_${R}.log 2>&1
    echo "done $out"
  done; done
}

# 架构与各任务【已有 run 一致】, 只对称扫 lr×R。
# Burgers(orig): hidden128 depth5 20k col
run_one train_pinn.py strata           burgers   "--hidden 128 --depth 5 --n_col 20000"
run_one train_pinn.py single_grassmann burgers   "--hidden 128 --depth 5 --n_col 20000"
# Burgers_g2: hidden96 depth4 20k col (与 runs2_burgers_g2 一致)
run_one train_pinn.py strata           burgersg2 "--hidden 96 --depth 4 --n_col 20000"
run_one train_pinn.py single_grassmann burgersg2 "--hidden 96 --depth 4 --n_col 20000"
# Poisson: hidden128 depth5 5k col (与 runs2_poisson 一致)
run_one train_poisson.py strata           poisson "--hidden 128 --depth 5 --n_col 5000"
run_one train_poisson.py single_grassmann poisson "--hidden 128 --depth 5 --n_col 5000"

echo "ALL SYMMETRIC GRID DONE"
