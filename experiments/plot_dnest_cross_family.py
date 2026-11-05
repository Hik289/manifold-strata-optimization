"""δ_nest 跨族对比图 (功能版, 科学内容正确; 发表级美化留 viz_expert)。

核心图 (论文贡献 2): δ_nest_norm vs 归一化学习进度, 每族一条线 + head-to-head 标注。
呈现规律: δ_nest 学习中【升高】(图)→坍缩→打平; 【低+平】(MNIST/PINN)→维持→不掉队。
判据是【形状/趋势】非绝对值 —— 用归一化学习进度做 x 轴凸显"升 vs 平"。

用法: python plot_dnest_cross_family.py
输入: dnest_cross_family.json
输出: figures/dnest_cross_family.png (+ .pdf)
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

# 族顺序 + 颜色 (功能版, viz_expert 会重配色)
order = ["graph_arxiv", "MNIST_MLP", "Fashion_MLP", "PINN_Burgers", "PINN_Poisson", "CNN_cifar"]
colors = {"graph_arxiv": "tab:red", "MNIST_MLP": "tab:green", "Fashion_MLP": "tab:olive",
          "PINN_Burgers": "tab:blue", "PINN_Poisson": "tab:cyan", "CNN_cifar": "tab:purple"}
labels = {"graph_arxiv": "graph (ogbn-arxiv)", "MNIST_MLP": "MNIST (MLP)",
          "Fashion_MLP": "Fashion (MLP)", "PINN_Burgers": "PINN (Burgers)",
          "PINN_Poisson": "PINN (Poisson)", "CNN_cifar": "CNN (CIFAR)"}


def smooth_bin(xs, ys, nb=12):
    """按归一化进度分箱平均, 让趋势清晰。"""
    xs, ys = np.array(xs), np.array(ys)
    edges = np.linspace(0, 1, nb + 1)
    bx, by = [], []
    for i in range(nb):
        m = (xs >= edges[i]) & (xs < edges[i + 1] + (1e-9 if i == nb - 1 else 0))
        if m.sum() > 0:
            bx.append((edges[i] + edges[i + 1]) / 2)
            by.append(ys[m].mean())
    return bx, by


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- 左: δ_nest vs 归一化学习进度 (核心) ---
for name in order:
    if name not in data:
        continue
    fam = data[name]
    traj = fam["dnest_trajectory"]
    if not traj:
        continue
    prog = [p["learn_progress"] for p in traj]
    dn = [p["dnest_norm"] for p in traj]
    bx, by = smooth_bin(prog, dn)
    h2h = {"tie": "tie", "tie(打平)": "tie",
           "tie(STRATA 0.983≈SG 0.980)": "tie", "tie(STRATA 0.899≈SG 0.903)": "tie",
           "STRATA (SG发散)": "STRATA(SG div.)"}.get(fam["head2head"], fam["head2head"])
    ax1.plot(bx, by, "-o", color=colors[name], ms=4, lw=2,
             label=f"{labels[name]} [{h2h}]")

ax1.set_xlabel("Normalized learning progress (0=start, 1=converged)")
ax1.set_ylabel("dnest_norm (indep-optimum nesting gap / random null)")
ax1.set_title("dnest dynamics: rise(collapse)->tie vs low-flat(persist)->no-dropout")
ax1.axhline(1.0, ls="--", c="gray", alpha=0.5, lw=1)
ax1.text(0.02, 1.02, "random null (no structure)", fontsize=8, c="gray")
ax1.legend(fontsize=8, loc="best")
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 1.05)

# --- 右: 前半 vs 后半 δ_nest (趋势条形, 凸显升 vs 平) ---
names_p = [n for n in order if n in data and data[n]["dnest_trajectory"]]
x = np.arange(len(names_p))
early = [data[n].get("dnest_early", np.nan) for n in names_p]
late = [data[n].get("dnest_late", np.nan) for n in names_p]
w = 0.35
ax2.bar(x - w / 2, early, w, label="early-half dnest", color="lightsteelblue")
ax2.bar(x + w / 2, late, w, label="late-half dnest", color="steelblue")
for i, n in enumerate(names_p):
    d = data[n].get("dnest_delta", 0)
    ax2.annotate(f"d={d:+.2f}", (i, max(early[i], late[i]) + 0.02), ha="center", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels([labels[n].split(" ")[0] for n in names_p], fontsize=9)
ax2.set_ylabel("dnest_norm mean")
ax2.set_title("dnest change over learning (rise=collapse; flat/drop=persist)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{BASE}/figures/dnest_cross_family.png", dpi=140, bbox_inches="tight")
plt.savefig(f"{BASE}/figures/dnest_cross_family.pdf", bbox_inches="tight")
print("写入 figures/dnest_cross_family.png + .pdf")

# 打印图注要点 (给 viz_expert + paper_writer)
print("\n=== 图注要点 (功能版, viz_expert 精修) ===")
print("核心规律: δ_nest 学习中升高(graph, 结构坍缩)→STRATA 打平;")
print("          δ_nest 低+平(MNIST/PINN, 结构维持)→STRATA 不掉队。")
print("关键: δ_nest 预测掉不掉队, 不预测胜负(MNIST 低δ_nest 但打平=反例;")
print("      PINN-Poisson 胜因 SG 发散非旗机制)。判据=趋势非绝对值。")
