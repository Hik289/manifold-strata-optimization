"""
独立核验 MLP (runs3_grid) + CNN (runs3_cnn) 族 — 从原始 log.jsonl 复算。
不信任 summary.json, 独立算末端100ep test_acc/val_acc 均值。
核验六族 head-to-head 的 vision 部分 + 对称公平选择 + eps=1e-8 崩溃证据保留。
"""
from pathlib import Path
import json, os
import numpy as np

BASE = str(Path(__file__).resolve().parents[1] / 'experiments')

def load(path):
    if not os.path.exists(path): return None
    rows = []
    for i, l in enumerate(open(path)):
        l = l.strip()
        if not l: continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError as e:
            import sys
            print(f"WARN bad line {i} in {path}: {e}", file=sys.stderr)
    return rows if rows else None

def end_mean(rows, key, n=100):
    v = [r[key] for r in rows if key in r and r[key] is not None]
    if not v: return None
    a = np.array(v[-n:], dtype=float)
    if not np.isfinite(a).all():
        return {"mean": None, "nonfinite": True, "n_finite": int(np.isfinite(a).sum())}
    return {"mean": float(a.mean()), "final": float(v[-1]), "min": float(a.min()),
            "max": float(a.max()), "n_epochs": len(rows)}

results = {"mlp": {}, "cnn": {}, "eps1e8_evidence": {}}

# ---- MLP grid (runs3_grid): mnist/fashion × strata/single_grassmann × lr{3e-4,1e-3,3e-3} × R64 ----
for ds in ["mnist", "fashion"]:
    results["mlp"][ds] = {}
    for m in ["strata", "single_grassmann", "adamw", "galore"]:
        for lr in ["3e-4", "1e-3", "3e-3"]:
            cell = f"{ds}__{m}__lr{lr}__R64"
            # training host: dir 结构
            p_dir = os.path.join(BASE, "runs3_grid", cell, "log.jsonl")
            rows = load(p_dir)
            if rows is None: continue
            va = end_mean(rows, "val_acc")
            ta = end_mean(rows, "test_acc")
            gn = end_mean(rows, "grad_norm")
            results["mlp"][ds][f"{m}_lr{lr}"] = {
                "val_acc_end100": va["mean"] if isinstance(va, dict) else None,
                "test_acc_end100": ta["mean"] if isinstance(ta, dict) else None,
                "gradnorm_max_end100": gn["max"] if isinstance(gn, dict) and "max" in gn else None,
                "n_epochs": len(rows),
            }

# ---- CNN grid (runs3_cnn): cifar10/100 × strata/single_grassmann × lr{3e-4,1e-3} × R64 ----
for ds in ["cifar10", "cifar100"]:
    results["cnn"][ds] = {}
    for m in ["strata", "single_grassmann"]:
        for lr in ["3e-4", "1e-3"]:
            cell = f"{ds}__{m}__lr{lr}__R64"
            p_dir = os.path.join(BASE, "runs3_cnn", cell, "log.jsonl")
            rows = load(p_dir)
            if rows is None: continue
            ta = end_mean(rows, "test_acc")
            va = end_mean(rows, "val_acc")
            results["cnn"][ds][f"{m}_lr{lr}"] = {
                "test_acc_end100": ta["mean"] if isinstance(ta, dict) else None,
                "val_acc_end100": va["mean"] if isinstance(va, dict) else None,
                "n_epochs": len(rows),
            }

# ---- eps=1e-8 崩溃证据 (runs3_eps1e-8_evidence) ----
for name in ["mnist__strata__lr3e-3__R64", "mnist__single_grassmann__lr3e-3__R64"]:
    sub = os.path.join(BASE, "runs3_eps1e-8_evidence", name, "log.jsonl")
    rows = load(sub)
    if rows is not None:
        va = end_mean(rows, "val_acc", n=100)
        allgn = [r["grad_norm"] for r in rows if r.get("grad_norm") is not None
                 and np.isfinite(r["grad_norm"])]
        results["eps1e8_evidence"][name] = {
            "max_gradnorm_alltraj": float(np.nanmax(allgn)) if allgn else None,
            "val_acc_end100": va["mean"] if isinstance(va, dict) else None,
            "n_epochs": len(rows), "path": sub,
        }

print(json.dumps(results, indent=2))
