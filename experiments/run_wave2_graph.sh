#!/bin/bash
# RUNNING-wave2 图任务: ogbn-arxiv / GraphSAGE, 1000 epochs, seed=42
# 有利任务族 (GNN 梯度谱低秩各向异性). 4 方法对比 + 全轨迹 δ_nest 诊断.
# ~2min/run, 单卡串行即可 (避开 manifoldgrass 占用的 GPU)
set -e
# Activate the project Python environment before running this launcher.
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p runs2_graph logs

GPU=${1:-2}
EP=1000
COMMON="--epochs $EP --lr 3e-3 --wd 0.0 --hidden 256 --n_layers 3 --dropout 0.5 --R 64 --seed 42 --warmup 20 --flag_log_every 10"

echo "using GPU $GPU"
# 串行跑 (每个 ~2min, 避免 GPU 竞争污染 timing)
CUDA_VISIBLE_DEVICES=$GPU python -u train_graph.py --method adamw $COMMON \
  --out runs2_graph/adamw > logs/graph_adamw.log 2>&1
echo "adamw done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_graph.py --method galore $COMMON \
  --galore_tau 200 --galore_scale 0.25 --out runs2_graph/galore > logs/graph_galore.log 2>&1
echo "galore done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_graph.py --method single_grassmann $COMMON \
  --xi 1e-3 --tau 50 --p 8 --a_perp 0.2 --out runs2_graph/single_grassmann > logs/graph_sg.log 2>&1
echo "single_grassmann done"
CUDA_VISIBLE_DEVICES=$GPU python -u train_graph.py --method strata $COMMON \
  --xi 1e-3 --tau 50 --p 8 --a_ks 1.0 0.7 0.4 --a_perp 0.2 --out runs2_graph/strata > logs/graph_strata.log 2>&1
echo "strata done"
echo "ALL 4 GRAPH RUNS DONE"
