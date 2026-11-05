#!/bin/bash
# wave3 MLP 对称公平网格: MNIST + Fashion, 5 方法 × lr{3e-3,1e-3,3e-4} × R{32,64}
# 每方法自身最优 (val=末端100 val_acc), seed=42, 1000ep. MLP ~50min/run.
# 用法: bash run_wave3_mlp_grid.sh <dataset> <gpu>
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs3_grid logs3
DS=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU
# 主敏感轴 = lr (崩溃来自 lr). 先扫 lr@R64, flag-like 再补 R32 于最优 lr。
LRS="3e-3 1e-3 3e-4"
COMMON="--dataset $DS --epochs 1000 --seed 42 --warmup 20 --wd 5e-4 --batch_size 128 --hidden 512 --flag_log_every 10"

run_cfg () {  # $1=method $2=lr $3=R
  local method=$1 lr=$2 R=$3
  local out=runs3_grid/${DS}__${method}__lr${lr}__R${R}
  [ -f $out/summary.json ] && { echo "skip $out"; return; }
  if [ "$method" = "strata" ]; then EXTRA="--xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2";
  elif [ "$method" = "galore" ]; then EXTRA="--galore_tau 200 --galore_scale 0.25";
  elif [ "$method" = "adamw" ]; then EXTRA="";
  else EXTRA="--xi 1e-3 --tau 50 --p 8 --a_perp 0.2"; fi
  python -u train_vision.py --method $method $COMMON --lr $lr --R $R $EXTRA \
    --out $out > logs3/grid_${DS}_${method}_${lr}_${R}.log 2>&1
  echo "done $out"
}

# 阶段1: 全方法 lr 扫 @ R=64
for method in adamw galore single_grassmann independent_grassmann strata; do
  for lr in $LRS; do run_cfg $method $lr 64; done
done
echo "ALL $DS MLP GRID (lr@R64) DONE"
