"""GraphSAGE (mean aggregator) for ogbn-arxiv, 纯 PyTorch 实现 (无 PyG)。

每层: h' = ReLU( BN( W_self·h + W_neigh·(D^-1 A·h) ) )
权重矩阵 W_self/W_neigh 是 2D 矩阵参数 (hidden×hidden), 走 STRATA 旗流形分支。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_graph import spmm


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_idx, adj_val, n):
        neigh = spmm(adj_idx, adj_val, n, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=256, n_cls=40, n_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.convs.append(SAGEConv(hidden, n_cls))
        self.dropout = dropout

    def forward(self, x, adj_idx, adj_val, n):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_idx, adj_val, n)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_idx, adj_val, n)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())
