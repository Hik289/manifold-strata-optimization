"""TRACE: MNIST MLP STRATA 爆炸点更新分量分解。

在每步分解 STRATA 更新为各层的 ‖D_perp‖ vs ‖D_k(Adam)‖ vs ‖D_total‖,
定位爆炸是否来自 D_perp 残差项 (my hypothesis) 还是 Adam 分支。
lr=3e-3 (已知崩), 跑 ~12 epoch 到爆炸点。
"""
from pathlib import Path
import json, sys, os
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as T
from model_cnn_mlp import MLP3
import strata_pkg.strata_adamw as S
from strata_pkg.strata_adamw import STRATAAdamW

dev = "cuda"
torch.manual_seed(42)
tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train = torchvision.datasets.MNIST(str(Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=tf)
tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2,
                                 generator=torch.Generator().manual_seed(42), drop_last=True)
model = MLP3(784, 512, 10).to(dev)
crit = nn.CrossEntropyLoss()
opt = STRATAAdamW(model.parameters(), lr=3e-3, ranks=[16, 32, 64], rhos=[0.999, 0.98, 0.9],
                 a_ks=[1.0, 0.7, 0.4], a_perp=0.2, xi=1e-3, tau=50, p=8, weight_decay=5e-4, seed=42)

# monkey-patch _matrix_update 记录各层分量
orig = S.STRATAAdamW._matrix_update
layer_stats = {}
def patched(self, state, G, group):
    transposed = state["transposed"]
    work = G.t().contiguous() if transposed else G
    U = state["U"]; slices = state["slices"]; ranks = group["ranks"]; K = len(ranks)
    eps = group["eps"]; a_perp = group["a_perp"]; a_ks = group["a_ks"]
    mw, d = work.shape
    # D_perp 分量
    G_perp = work - (work @ U) @ U.t()
    denom = torch.linalg.norm(G_perp) / (mw * d) ** 0.5 + eps
    Dperp = a_perp * (G_perp / denom)
    r = orig(self, state, G, group)
    # D_k Adam 分量 (总 - 残差)
    key = tuple(G.shape)
    layer_stats.setdefault(key, []).append(
        (float(Dperp.norm()), float(r.norm()), float(G.norm()), mw, d))
    return r
S.STRATAAdamW._matrix_update = patched

print("ep step | shape | ||D_perp|| ||D_total|| ||G|| | Dperp/Dtot")
for ep in range(14):
    layer_stats.clear()
    tls = 0; n = 0
    gnmax = 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        gnmax = max(gnmax, float(gn))
        opt.step(); tls += loss.item(); n += 1
    # 每 epoch 报一次各层首步分量
    line = f"ep{ep:<3} loss={tls/n:.3g} gnmax={gnmax:.3g}"
    for key, vals in layer_stats.items():
        dp, dt, g, mw, d = vals[0]
        line += f" | {mw}x{d}: Dp={dp:.3g} Dt={dt:.3g} r={dp/(dt+1e-12):.2f}"
    print(line, flush=True)
