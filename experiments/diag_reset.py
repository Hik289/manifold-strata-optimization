import torch, sys
sys.path.insert(0, ".")
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
from model import CompactViT
import strata_pkg.strata_adamw as S

dev = "cuda"
model = CompactViT(num_classes=100).to(dev)
x = torch.randn(64, 3, 32, 32, device=dev)
y = torch.randint(0, 100, (64,), device=dev)
crit = torch.nn.CrossEntropyLoss()
# tau=3 让 flag update 频繁, 观察 reset 后一步的 H 量级. a_perp=0 排除残差干扰
opt = S.STRATAAdamW(model.parameters(), lr=3e-4, ranks=[16, 32, 64], rhos=[0.999, 0.98, 0.9],
                    a_ks=[1.0, 0.7, 0.4], a_perp=0.0, xi=1e-3, tau=3, p=8, weight_decay=0.05, seed=42)

for step in range(12):
    opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward()
    opt.step()
    for p in model.parameters():
        st = opt.state.get(p, {})
        if "U" in st and tuple(p.shape) == (768, 256):
            c = st["c"]
            Mn = [round(float(m.norm()), 2) for m in st["M"]]
            Vn = [round(float(v.norm()), 4) for v in st["V"]]
            just_reset = (st["t"] - 1) % 3 == 0
            print(f"step{step} t={st['t']} c={c} reset_last={just_reset} "
                  f"|M|={Mn} |V|={Vn} gW={float(p.grad.norm()):.3g}")
            break
