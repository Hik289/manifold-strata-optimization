"""
补充诊断: 区分 '嵌套且各时间尺度真正不同' (旗 > 单一 Grassmann)
vs '嵌套但各时间尺度几乎相同' (旗退化为 Grassmann, 无收益)。

关键量:
 - 跨时间尺度 同秩重叠矩阵 (r_1, r_2, r_3): 若全=1 则各尺度相同->旗无意义;
   若内层高、外层低 -> 内核共享 + 外层分化 (旗结构的理想证据)
 - '身份对照': 同一时间尺度用两个不相交 minibatch 半批 估计的 top-r 重叠
   (自一致上界; 跨尺度重叠应显著低于此才算真正分化)
seed=42.
"""
from __future__ import annotations
from pathlib import Path
import json, sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from strata_pkg import diagnostics as diag

SEED = 42; DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = str(Path(__file__).resolve().parents[2] / 'data/raw')
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


class SmallMLP(nn.Module):
    def __init__(s, d_in=3072, h1=256, h2=128, n=10):
        super().__init__(); s.fc1=nn.Linear(d_in,h1); s.fc2=nn.Linear(h1,h2); s.fc3=nn.Linear(h2,n)
    def forward(s,x): return s.fc3(F.relu(s.fc2(F.relu(s.fc1(x)))))


def load_cifar():
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", split="train")
    X = torch.tensor(np.stack([np.array(im) for im in ds["img"]]).astype(np.float32)/255.).reshape(len(ds),-1)
    X = (X - X.mean(0,keepdim=True))/(X.std(0,keepdim=True)+1e-6)
    return X.to(DEV), torch.tensor(ds["label"]).to(DEV)


def overlap(A, B):
    ang = diag.principal_angles(A, B)
    return float(torch.cos(ang).pow(2).mean())


def main():
    rhos = (0.999, 0.98, 0.9); ranks = (8, 16, 32); K = 3
    X, y = load_cifar()
    model = SmallMLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    W = model.fc1.weight
    n = min(W.shape)  # 256
    C = [torch.zeros(n, n, device=DEV) for _ in range(K)]
    steps = 1200
    for t in range(steps):
        idx = torch.randint(0, X.size(0), (128,), device=DEV)
        opt.zero_grad(); loss = F.cross_entropy(model(X[idx]), y[idx]); loss.backward()
        G = W.grad.detach()
        # fc1: (256,3072) -> m=256<n=3072 -> 左侧 GG^T (256x256)
        Ct = G @ G.t() if W.shape[0] <= W.shape[1] else G.t() @ G
        for k in range(K): C[k].mul_(rhos[k]).add_(Ct, alpha=(1-rhos[k]))
        opt.step()

    # 身份对照: 训练后固定权重, 用两个不相交 minibatch 估计短尺度 top-r 重叠
    model.eval()
    def batch_cov(seed_off):
        g = torch.Generator(device=DEV); g.manual_seed(SEED + seed_off)
        acc = torch.zeros(n, n, device=DEV)
        for _ in range(40):
            idx = torch.randint(0, X.size(0), (128,), device=DEV, generator=g)
            model.zero_grad(); L = F.cross_entropy(model(X[idx]), y[idx]); L.backward()
            G = W.grad.detach(); acc += (G @ G.t())
        return acc / 40
    Ca = batch_cov(0); Cb = batch_cov(9999)

    # 跨时间尺度 同秩重叠矩阵
    res = {"rhos": list(rhos), "ranks": list(ranks), "n_cov": n}
    for ri, r in enumerate(ranks):
        Vs = [diag.top_r_subspace(C[k], r) for k in range(K)]
        M = [[round(overlap(Vs[i], Vs[j]), 4) for j in range(K)] for i in range(K)]
        res[f"cross_timescale_overlap_rank{r}"] = M
    # 身份自一致上界 (同尺度不同批 top-r 重叠) — 各 rank
    res["identity_selfconsistency"] = {
        f"rank{r}": round(overlap(diag.top_r_subspace(Ca, r), diag.top_r_subspace(Cb, r)), 4)
        for r in ranks}
    # 跨相邻尺度 outer-rank 重叠 vs 自一致上界: 分化证据
    adj_outer = overlap(diag.top_r_subspace(C[1], ranks[-1]), diag.top_r_subspace(C[2], ranks[-1]))
    res["adjacent_outer_overlap_rankR"] = round(float(adj_outer), 4)
    print(json.dumps(res, indent=2))
    json.dump(res, open(str(Path(__file__).resolve().parents[2] / 'data/grad_struct_supp.json'), "w"), indent=2)


if __name__ == "__main__":
    main()
