"""验证 eps 修复: 512x512 层 Adam 分支爆炸是否因 eps 太小 (V≈0 时 H=M/eps 爆)。
对比原 eps=1e-8 vs 修复方案。"""
from pathlib import Path
import sys; sys.path.insert(0, ".")
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from model_cnn_mlp import MLP3
from strata_pkg.strata_adamw import STRATAAdamW

dev = "cuda"
tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train = torchvision.datasets.MNIST(str(Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=tf)

def run(eps, tag, steps=600):
    torch.manual_seed(42)
    tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2,
                                     generator=torch.Generator().manual_seed(42), drop_last=True)
    model = MLP3(784, 512, 10).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = STRATAAdamW(model.parameters(), lr=3e-3, ranks=[16, 32, 64], rhos=[0.999, 0.98, 0.9],
                     a_ks=[1.0, 0.7, 0.4], a_perp=0.2, xi=1e-3, tau=50, p=8,
                     eps=eps, weight_decay=5e-4, seed=42)
    step = 0; maxgn = 0
    for ep in range(6):
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
            maxgn = max(maxgn, gn)
            opt.step(); step += 1
            if step >= steps:
                break
        if step >= steps:
            break
    # eval
    from torch.utils.data import DataLoader
    test = torchvision.datasets.MNIST(str(Path(__file__).resolve().parents[1] / 'data'), train=False, download=False, transform=tf)
    tel = DataLoader(test, batch_size=512)
    model.eval(); cor = tot = 0
    with torch.no_grad():
        for x, y in tel:
            x, y = x.to(dev), y.to(dev)
            cor += (model(x).argmax(1) == y).sum().item(); tot += y.numel()
    print(f"[{tag}] eps={eps:.0e}: {steps}步 maxgn={maxgn:.3g} test_acc={cor/tot:.4f} finite_loss={torch.isfinite(loss).item()}", flush=True)

run(1e-8, "原始")
run(1e-6, "eps=1e-6")
run(1e-4, "eps=1e-4")
run(1e-3, "eps=1e-3")
