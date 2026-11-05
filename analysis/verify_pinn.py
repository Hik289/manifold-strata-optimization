"""
独立核验 PINN 族 (runs2_grid_eps1e6) — Data Scientist ANALYSIS 关口
从原始 log.jsonl 独立复算末端100ep rel_l2 均值, 不信任 summary.json。
核验:
 - 六族 head-to-head 的 PINN 部分 (Burgers/Burgers_g2/Poisson)
 - 每方法各自最优 cell 是否真为验证集最优 (对称公平)
 - A5 两次最优点消融 (Γᵃ=0 不退化)
 - 不利结果是否保留 (Poisson 高lr 发散/NaN, SG Poisson NaN)
输出 JSON 到 stdout。
"""
from pathlib import Path
import json, os, glob, math
import numpy as np

ROOT = str(Path(__file__).resolve().parents[1] / 'experiments/runs2_grid_eps1e6')
ROOT_RUNS2GRID = str(Path(__file__).resolve().parents[1] / 'experiments/runs2_grid')

def load_log(cell):
    p = os.path.join(ROOT, cell, "log.jsonl")
    if not os.path.exists(p):
        p2 = os.path.join(ROOT_RUNS2GRID, cell, "log.jsonl")
        if os.path.exists(p2): p = p2
        else: return None
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def end_mean(rows, key, n=100):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals: return None
    tail = vals[-n:]
    # 检查 NaN/inf
    arr = np.array(tail, dtype=float)
    if np.any(~np.isfinite(arr)):
        return {"mean": None, "nan_or_inf": True, "n_finite": int(np.isfinite(arr).sum())}
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()),
            "std": float(arr.std()), "final": float(vals[-1]), "n_epochs": len(rows)}

def diverged(rows):
    """判定是否发散: rel_l2 出现 NaN 或末端>0.5 且 grad爆"""
    rl = [r.get("rel_l2") for r in rows if "rel_l2" in r]
    if not rl: return "no_rel_l2"
    arr = np.array([x if x is not None else float('nan') for x in rl], dtype=float)
    if np.any(np.isnan(arr)): return "NaN"
    if arr[-1] > 0.5: return f"high_end={arr[-1]:.3f}"
    return "ok"

results = {"pinn_grid": {}, "A5": {}, "optimal_selection": {}}

tasks = ["burgers", "burgersg2", "poisson"]
methods = ["strata", "single_grassmann", "adamw", "galore"]
lrs = ["5e-3", "1e-3", "5e-4"]
Rs = ["32", "64"]

# 全网格复算
for task in tasks:
    results["pinn_grid"][task] = {}
    for method in methods:
        results["pinn_grid"][task][method] = {}
        for lr in lrs:
            for R in Rs:
                cell = f"{task}__{method}__lr{lr}__R{R}"
                rows = load_log(cell)
                if rows is None:
                    continue
                em = end_mean(rows, "rel_l2")
                dv = diverged(rows)
                results["pinn_grid"][task][method][f"lr{lr}_R{R}"] = {
                    "end100_rel_l2": em["mean"] if isinstance(em, dict) and "mean" in em else None,
                    "raw": em, "diverge_flag": dv, "n_epochs": len(rows)
                }

# 每方法各自最优 cell (验证集=末端100 loss_pde 最小, 与 analysis 协议一致; 但也报 rel_l2)
def best_cell(task, method):
    best = None
    for lr in lrs:
        for R in Rs:
            cell = f"{task}__{method}__lr{lr}__R{R}"
            rows = load_log(cell)
            if rows is None: continue
            # 验证信号: loss_pde 末端100均值 (排除退化 val_pde<1e-7 且 rel_l2>0.5, 与 analysis 一致)
            vp = end_mean(rows, "loss_pde")
            rl = end_mean(rows, "rel_l2")
            if not isinstance(vp, dict) or vp.get("mean") is None: continue
            if not isinstance(rl, dict) or rl.get("mean") is None: continue
            val_pde = vp["mean"]; rel = rl["mean"]
            if val_pde < 1e-7 and rel > 0.5:  # 退化排除
                continue
            score = val_pde
            if best is None or score < best["val_pde"]:
                best = {"cell": f"lr{lr}_R{R}", "val_pde": val_pde, "rel_l2": rel}
    return best

for task in tasks:
    results["optimal_selection"][task] = {}
    for method in ["strata", "single_grassmann"]:
        results["optimal_selection"][task][method] = best_cell(task, method)

# A5 消融
a5_cells = {
    "burgers_full_lr5e-3_R64": ("burgers__strata__lr5e-3__R64", ROOT),
    "burgers_A5_gamma0": ("A5_gamma0_optimal_lr5e-3_R64", ROOT),
    "burgers_A5_gamma0_runs2grid": ("A5_gamma0_optimal_lr5e-3_R64", ROOT_RUNS2GRID),
    "poisson_full_lr5e-4_R32": ("poisson__strata__lr5e-4__R32", ROOT),
    "poisson_A5_gamma0": ("A5_poisson_gamma0_lr5e-4_R32", ROOT),
}
for label, (cell, root) in a5_cells.items():
    p = os.path.join(root, cell, "log.jsonl")
    if not os.path.exists(p):
        results["A5"][label] = {"exists": False, "path": p}
        continue
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    em = end_mean(rows, "rel_l2")
    results["A5"][label] = {"exists": True, "end100_rel_l2": em["mean"] if isinstance(em,dict) else None,
                             "n_epochs": len(rows), "path": p}

print(json.dumps(results, indent=2))
