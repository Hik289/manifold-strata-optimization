#!/bin/bash
# RUNNING-wave3: CNN(ResNet-18)/MLP benchmark, 1000ep, seed=42
# CNN: cifar10/cifar100 (conv-2D 化); MLP: mnist/fashion (天然 2D)
# 5 方法 × 4 数据集 = 20 runs。CNN ~7h/run, MLP ~50min/run。跨 3 GPU 并行。
# 用法: bash run_wave3_vision.sh <dataset> <gpu>
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs3 logs3
DS=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU

if [ "$DS" = "cifar10" ] || [ "$DS" = "cifar100" ]; then
  EXTRA="--batch_size 128 --lr 1e-3 --wd 5e-4 --warmup 20"
else
  EXTRA="--batch_size 128 --lr 1e-3 --wd 5e-4 --warmup 20 --hidden 512"
fi
COMMON="--dataset $DS --epochs 1000 --R 64 --seed 42 --flag_log_every 10 $EXTRA"

python -u train_vision.py --method adamw $COMMON --out runs3/${DS}_adamw > logs3/${DS}_adamw.log 2>&1
echo "$DS adamw done"
python -u train_vision.py --method galore $COMMON --galore_tau 200 --galore_scale 0.25 --out runs3/${DS}_galore > logs3/${DS}_galore.log 2>&1
echo "$DS galore done"
python -u train_vision.py --method single_grassmann $COMMON --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 --out runs3/${DS}_single_grassmann > logs3/${DS}_single_grassmann.log 2>&1
echo "$DS single_grassmann done"
python -u train_vision.py --method independent_grassmann $COMMON --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 --out runs3/${DS}_independent_grassmann > logs3/${DS}_independent_grassmann.log 2>&1
echo "$DS independent_grassmann done"
python -u train_vision.py --method strata $COMMON --xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2 --out runs3/${DS}_strata > logs3/${DS}_strata.log 2>&1
echo "$DS strata done"
echo "ALL $DS DONE"
