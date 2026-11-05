#!/bin/bash
# MLP 优先并行: STRATA + Single-Grassmann 全 lr cell 分散到 3 卡 (implementation 要求)。
# 12 优先 cell (2 数据集 × 2 方法 × 3 lr) round-robin over GPU 0/1/2。
# eps=1e-6 (默认已改)。galore 剩余 cell 放最后 (另跑)。
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs3_grid logs3
# Activate the project Python environment before running this launcher.

COMMON="--epochs 1000 --seed 42 --warmup 20 --wd 5e-4 --batch_size 128 --hidden 512 --flag_log_every 10 --R 64"

# 生成 (dataset, method, lr) 任务列表
TASKS=()
for ds in mnist fashion; do
  for m in strata single_grassmann; do
    for lr in 3e-3 1e-3 3e-4; do
      TASKS+=("$ds $m $lr")
    done
  done
done

run_cell () {
  local ds=$1 m=$2 lr=$3 gpu=$4
  local out=runs3_grid/${ds}__${m}__lr${lr}__R64
  [ -f $out/summary.json ] && { echo "skip $out"; return; }
  if [ "$m" = "strata" ]; then EX="--xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2"; else EX="--xi 1e-3 --tau 50 --p 8 --a_perp 0.2"; fi
  CUDA_VISIBLE_DEVICES=$gpu python -u train_vision.py --method $m --dataset $ds $COMMON --lr $lr $EX \
    --out $out > logs3/par_${ds}_${m}_${lr}.log 2>&1
  echo "done $out (gpu$gpu)"
}

# 每张卡一个 worker, 从共享队列取任务 (用文件锁模拟: 按 index 取模分配)
worker () {
  local gpu=$1
  local i=$gpu
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
echo "ALL MLP PRIORITY (STRATA+SG) DONE"
