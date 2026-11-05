#!/bin/bash
# REVISION 多 seed: CIFAR-100 (vision 代表) STRATA+SG 最优 cell(lr1e-3) × seeds {0,1}
# seed 42 已有(runs3_cnn/)。补 seed 0,1。4 runs round-robin 4 卡。
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs_revision_seed logs_revision
# Activate the project Python environment before running this launcher.
COMMON="--dataset cifar100 --epochs 1000 --warmup 20 --wd 5e-4 --batch_size 128 --R 64 --lr 1e-3 --flag_log_every 10"

TASKS=()
for seed in 0 1; do
  for m in strata single_grassmann; do
    TASKS+=("$seed $m")
  done
done
run_cell () {
  local seed=$1 m=$2 gpu=$3
  local out=runs_revision_seed/cifar100__${m}__lr1e-3__R64__seed${seed}
  [ -f $out/summary.json ] && { echo "skip $out"; return; }
  if [ "$m" = "strata" ]; then EX="--xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2"; else EX="--xi 1e-3 --tau 50 --p 8 --a_perp 0.2"; fi
  CUDA_VISIBLE_DEVICES=$gpu python -u train_vision.py --method $m $COMMON --seed $seed $EX \
    --out $out > logs_revision/c100_${m}_seed${seed}.log 2>&1
  echo "done $out (gpu$gpu)"
}
# 4 runs, 4 卡各一
i=0
for t in "${TASKS[@]}"; do
  read seed m <<< "$t"
  run_cell $seed $m $i &
  i=$((i+1))
done
wait
echo "ALL CIFAR-100 MULTISEED DONE"
