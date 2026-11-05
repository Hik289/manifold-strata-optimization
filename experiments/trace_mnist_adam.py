"""TRACE 2: 512x512 层 Adam 分支爆炸——查 M/V/Hhat/c 哪个爆。"""
from pathlib import Path
import sys; sys.path.insert(0, ".")
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from model_cnn_mlp import MLP3
import strata_pkg.strata_adamw as S
from strata_pkg.strata_adamw import STRATAAdamW

dev = "cuda"; torch.manual_seed(42)
tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train = torchvision.datasets.MNIST(str(Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=tf)
tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2,
                                 generator=torch.Generator().manual_seed(42), drop_last=True)
model = MLP3(784, 512, 10).to(dev)
crit = nn.CrossEntropyLoss()
opt = STRATAAdamW(model.parameters(), lr=3e-3, ranks=[16, 32, 64], rhos=[0.999, 0.98, 0.9],
                 a_ks=[1.0, 0.7, 0.4], a_perp=0.2, xi=1e-3, tau=50, p=8, weight_decay=5e-4, seed=42)

# 找到 512x512 参数
target = None
for p in model.parameters():
    if tuple(p.shape) == (512, 512):
        target = p; break

step = 0
for ep in range(4):
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        step += 1
        st = opt.state[target]
        if "M" in st and (step % 50 == 0 or step < 5):
            Mn = [round(float(m.norm()), 3) for m in st["M"]]
            Vn = [float(v.norm()) for v in st["V"]]
            Vmin = [float(v.min()) for v in st["V"]]
            cM = st["c_M"]; cV = st["c_V"]
            # 复算 Hhat 每块
            Hn = []
            for k in range(3):
                Mhat = st["M"][k] / (1 - 0.9 ** cM[k])
                Vhat = st["V"][k] / (1 - 0.999 ** cV[k])
                H = Mhat / (torch.sqrt(Vhat) + 1e-8)
                Hn.append(round(float(H.norm()), 2))
            print(f"step{step:<5} ep{ep} |M|={Mn} |V|={[round(v,5) for v in Vn]} Vmin={[round(v,8) for v in Vmin]} cM={cM} cV={cV} |Hhat|={Hn}", flush=True)
        if step > 200:
            break
    if step > 200:
        break
