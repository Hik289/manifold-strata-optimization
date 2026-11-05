#!/bin/bash
# CNN 第三族 scoped 版 (implementation 商定): STRATA + Single-Grassmann, eps=1e-6, conv-2D 化。
# CIFAR-10/100 × lr{3e-4,1e-3} (避开 lr3e-3 高 lr 不稳区, MLP 最优在 lr3e-4)。
# AdamW/GaLore 参考已有(runs3/, eps 不敏感)。8 cell round-robin 3 卡并行。
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs3_cnn logs3
# Activate the project Python environment before running this launcher.

COMMON="--epochs 1000 --seed 42 --warmup 20 --wd 5e-4 --batch_size 128 --R 64 --flag_log_every 10"

TASKS=()
for ds in cifar10 cifar100; do
  for m in strata single_grassmann; do
    for lr in 3e-4 1e-3; do
      TASKS+=("$ds $m $lr")
    done
  done
done

run_cell () {
  local ds=$1 m=$2 lr=$3 gpu=$4
  local out=runs3_cnn/${ds}__${m}__lr${lr}__R64
  [ -f $out/summary.json ] && { echo "skip $out"; return; }
  if [ "$m" = "strata" ]; then EX="--xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2"; else EX="--xi 1e-3 --tau 50 --p 8 --a_perp 0.2"; fi
  CUDA_VISIBLE_DEVICES=$gpu python -u train_vision.py --method $m --dataset $ds $COMMON --lr $lr $EX \
    --out $out > logs3/cnn_${ds}_${m}_${lr}.log 2>&1
  echo "done $out (gpu$gpu)"
}

worker () {
  local gpu=$1 i=$1
  while [ $i -lt ${#TASKS[@]} ]; do
    read ds m lr <<< "${TASKS[$i]}"
    run_cell $ds $m $lr $gpu
    i=$((i + 3))
  done
  echo "worker gpu$gpu done"
}

worker 0 &
worker 1 &
worker 2 &
wait
echo "ALL CNN SCOPED DONE"
