"""ogbn-arxiv 数据加载 (从 raw CSV, 无 ogb/PyG 依赖)。

数据: ./data/raw/ogb/ogbn_arxiv/raw/*.csv.gz
- node-feat.csv.gz: (169343, 128) 节点特征
- edge.csv.gz: (1166243, 2) 有向边
- node-label.csv.gz: (169343, 1) 40 类
- split/time/{train,valid,test}.csv.gz: 官方时间划分索引

返回 GPU 张量 + 对称归一化邻接 (GraphSAGE mean 聚合用 D^-1 A)。
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import torch


def _load_csv_gz(path, dtype=np.float32):
    with gzip.open(path, "rt") as f:
        arr = np.loadtxt(f, delimiter=",", dtype=dtype)
    return arr


def load_arxiv(data_root, device="cuda", cache=True):
    """返回 dict: x, y, train_idx, val_idx, test_idx, adj_indices, adj_values, n, feat_dim, n_cls。"""
    raw = os.path.join(data_root, "raw/ogb/ogbn_arxiv/raw")
    split = os.path.join(data_root, "raw/ogb/ogbn_arxiv/split/time")
    cache_path = os.path.join(data_root, "arxiv_cache.pt")
    if cache and os.path.exists(cache_path):
        d = torch.load(cache_path, map_location="cpu")
    else:
        x = _load_csv_gz(os.path.join(raw, "node-feat.csv.gz"), np.float32)   # (N,128)
        y = _load_csv_gz(os.path.join(raw, "node-label.csv.gz"), np.int64).reshape(-1)  # (N,)
        edge = _load_csv_gz(os.path.join(raw, "edge.csv.gz"), np.int64)       # (E,2)
        train_idx = _load_csv_gz(os.path.join(split, "train.csv.gz"), np.int64).reshape(-1)
        val_idx = _load_csv_gz(os.path.join(split, "valid.csv.gz"), np.int64).reshape(-1)
        test_idx = _load_csv_gz(os.path.join(split, "test.csv.gz"), np.int64).reshape(-1)
        d = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "edge": torch.from_numpy(edge),
            "train_idx": torch.from_numpy(train_idx),
            "val_idx": torch.from_numpy(val_idx),
            "test_idx": torch.from_numpy(test_idx),
        }
        if cache:
            torch.save(d, cache_path)

    x = d["x"]
    y = d["y"]
    edge = d["edge"]
    n = x.shape[0]

    # 无向化 + 自环, 构造 D^-1 A (mean 聚合)
    src = torch.cat([edge[:, 0], edge[:, 1], torch.arange(n)])
    dst = torch.cat([edge[:, 1], edge[:, 0], torch.arange(n)])
    idx = torch.stack([dst, src], dim=0)  # (2, E'), 行=目标 列=源 → A[dst,src]
    deg = torch.zeros(n).scatter_add_(0, dst, torch.ones(dst.shape[0]))
    vals = 1.0 / deg[dst].clamp(min=1.0)  # D^-1 归一

    out = {
        "x": x.to(device),
        "y": y.to(device),
        "train_idx": d["train_idx"].to(device),
        "val_idx": d["val_idx"].to(device),
        "test_idx": d["test_idx"].to(device),
        "adj_idx": idx.to(device),
        "adj_val": vals.to(device),
        "n": n, "feat_dim": x.shape[1], "n_cls": int(y.max().item() + 1),
    }
    return out


def spmm(adj_idx, adj_val, m, x):
    """稀疏 D^-1 A @ x。adj_idx (2,E) [dst,src], adj_val (E,), x (n,f)。"""
    dst, src = adj_idx[0], adj_idx[1]
    msg = x[src] * adj_val.unsqueeze(1)
    out = torch.zeros(m, x.shape[1], dtype=x.dtype, device=x.device)
    out.index_add_(0, dst, msg)
    return out
