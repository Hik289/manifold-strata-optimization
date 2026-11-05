"""判别实验: D_perp 假设(implementation) vs eps 假设(我的 TRACE)。
4 组对照, 各 600 步 MNIST STRATA lr=3e-3, 报 maxgn + test_acc + 是否崩。
- baseline: a_perp=0.2, eps=1e-8  (已知崩)
- INTERVENE a_perp=0, eps=1e-8    (测 D_perp: 若修好=D_perp因果)
- eps=1e-6, a_perp=0.2            (测 eps: 若修好=eps因果)
- eps=1e-6, a_perp=0             (两个都关, 对照)
"""
from pathlib import Path
import sys; sys.path.insert(0, ".")
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from model_cnn_mlp import MLP3
from strata_pkg.strata_adamw import STRATAAdamW

dev = "cuda"
tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train = torchvision.datasets.MNIST(str(Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=tf)
test = torchvision.datasets.MNIST(str(Path(__file__).resolve().parents[1] / 'data'), train=False, download=False, transform=tf)
tel = torch.utils.data.DataLoader(test, batch_size=512)

def run(a_perp, eps, tag, steps=600):
    torch.manual_seed(42)
    tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2,
                                     generator=torch.Generator().manual_seed(42), drop_last=True)
    model = MLP3(784, 512, 10).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = STRATAAdamW(model.parameters(), lr=3e-3, ranks=[16, 32, 64], rhos=[0.999, 0.98, 0.9],
                     a_ks=[1.0, 0.7, 0.4], a_perp=a_perp, xi=1e-3, tau=50, p=8,
                     eps=eps, weight_decay=5e-4, seed=42)
    step = 0; maxgn = 0
    for ep in range(6):
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward()
            maxgn = max(maxgn, float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)))
            opt.step(); step += 1
            if step >= steps: break
        if step >= steps: break
    model.eval(); cor = tot = 0
    with torch.no_grad():
        for x, y in tel:
            x, y = x.to(dev), y.to(dev)
            cor += (model(x).argmax(1) == y).sum().item(); tot += y.numel()
    crashed = maxgn > 100 or not torch.isfinite(loss).item()
    print(f"[{tag}] a_perp={a_perp} eps={eps:.0e}: maxgn={maxgn:.3g} test={cor/tot:.4f} {'CRASH' if crashed else 'OK'}", flush=True)

run(0.2, 1e-8, "baseline(原)")
run(0.0, 1e-8, "INTERVENE a_perp=0 (测D_perp)")
run(0.2, 1e-6, "eps=1e-6 (测eps)")
run(0.0, 1e-6, "both off")
