"""δ_nest 跨族对比图 — EPOCH 轴 (诚实版, DS 核验后) + learn_progress 对照。

DS 核验: learn_progress 分箱是伪影(前期即近收敛任务的 epoch 挤在 progress≈1)。
本图以 epoch 轴为主(诚实), learn_progress 轴为对照(标 caveat)。
诚实结论: graph 唯一强单调坍缩; 但 Burgers/CNN/Poisson δ_nest 也升却打平 → 无干净预测器。
"""
from pathlib import Path
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = str(Path(__file__).resolve().parents[1] / 'experiments')
data = json.load(open(f"{BASE}/dnest_cross_family.json"))
os.makedirs(f"{BASE}/figures", exist_ok=True)

order = ["graph_arxiv", "MNIST_MLP", "Fashion_MLP", "PINN_Burgers", "PINN_Poisson", "CNN_cifar"]
colors = {"graph_arxiv": "tab:red", "MNIST_MLP": "tab:green", "Fashion_MLP": "tab:olive",
          "PINN_Burgers": "tab:blue", "PINN_Poisson": "tab:cyan", "CNN_cifar": "tab:purple"}
labels = {"graph_arxiv": "graph (ogbn-arxiv)", "MNIST_MLP": "MNIST (MLP)",
          "Fashion_MLP": "Fashion (MLP)", "PINN_Burgers": "PINN (Burgers)",
          "PINN_Poisson": "PINN (Poisson)", "CNN_cifar": "CNN (CIFAR)"}
h2h_short = {"tie": "tie", "tie(打平)": "tie", "tie(STRATA 0.983≈SG 0.980)": "tie",
             "tie(STRATA 0.899≈SG 0.903)": "tie", "tie(STRATA 0.940≈SG 0.940)": "tie",
             "STRATA (SG发散)": "STRATA(SGdiv)"}


def normepoch(traj):
    eps = np.array([p["epoch"] for p in traj], float)
    dn = np.array([p["dnest_norm"] for p in traj], float)
    if eps.max() > 0:
        eps = eps / eps.max()
    return eps, dn


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左: EPOCH 轴 (诚实主图)
for name in order:
    if name not in data:
        continue
    traj = data[name]["dnest_trajectory"]
    if not traj:
        continue
    ex, dn = normepoch(traj)
    h = h2h_short.get(data[name]["head2head"], data[name]["head2head"])
    ax1.plot(ex, dn, "-", color=colors[name], lw=1.8, alpha=0.85,
             label=f"{labels[name]} [{h}]")
ax1.set_xlabel("epoch (normalized to [0,1])")
ax1.set_ylabel("dnest_norm (indep-optimum nesting gap / random null)")
ax1.set_title("EPOCH axis (honest): 3 families rise monotonically (graph/Burgers/Poisson, Spearman .82-.99);\ngraph unique only in LOW start (0.05) + collapse-from-nested; all rise yet TIE -> NO clean predictor")
ax1.axhline(1.0, ls="--", c="gray", alpha=0.5, lw=1)
ax1.legend(fontsize=7.5, loc="best")
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 1.05)

# 右: EPOCH 前半 vs 后半 (诚实趋势)
names_p = [n for n in order if n in data and data[n]["dnest_trajectory"]]
x = np.arange(len(names_p))
early, late = [], []
for n in names_p:
    traj = data[n]["dnest_trajectory"]
    maxep = max(p["epoch"] for p in traj)
    e = [p["dnest_norm"] for p in traj if p["epoch"] < maxep * 0.5]
    l = [p["dnest_norm"] for p in traj if p["epoch"] >= maxep * 0.5]
    early.append(np.mean(e) if e else np.nan)
    late.append(np.mean(l) if l else np.nan)
w = 0.35
ax2.bar(x - w / 2, early, w, label="epoch early-half", color="lightsteelblue")
ax2.bar(x + w / 2, late, w, label="epoch late-half", color="steelblue")
for i in range(len(names_p)):
    d = late[i] - early[i]
    ax2.annotate(f"{d:+.2f}", (i, max(early[i], late[i]) + 0.02), ha="center", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels([labels[n].split(" ")[0] for n in names_p], fontsize=9, rotation=15)
ax2.set_ylabel("dnest_norm mean")
ax2.set_title("EPOCH-axis early vs late (multiple families rise, all tie)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{BASE}/figures/dnest_epoch_honest.png", dpi=140, bbox_inches="tight")
plt.savefig(f"{BASE}/figures/dnest_epoch_honest.pdf", bbox_inches="tight")
print("写入 figures/dnest_epoch_honest.png + .pdf")
print("\n诚实结论: 三族强单调升(graph/Burgers/Poisson, Spearman .82-.99); graph 唯一在'低起点0.05+从良好嵌套坍缩'口径")
print("多族升却全 tie => δ_nest 趋势与胜负无稳健预测关系 (与 A5 机制负结果一致)")
