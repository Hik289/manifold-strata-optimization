"""
独立核验 δ_nest 框架 — 从原始 STRATA run 的 flag_diag.indep_dnest_norm_mean
独立复算 early/late, 与 dnest_cross_family.json 对比。
early = 前半均值, late = 后半均值 (与 consolidate 协议一致性检查)。
在各源机器分别跑 (PINN@training host, MLP/CNN@training host)。
"""
from pathlib import Path
import json, os, sys
import numpy as np

BASE = str(Path(__file__).resolve().parents[1] / 'experiments')

def load(path):
    if not os.path.exists(path): return None
    rows = []
    for l in open(path):
        l = l.strip()
        if not l: continue
        try: rows.append(json.loads(l))
        except: pass
    return rows or None

def dnest_series(rows):
    """提取每 epoch 的 indep_dnest_norm_mean (来自 flag_diag)。"""
    s = []
    for r in rows:
        fd = r.get("flag_diag") or {}
        v = fd.get("indep_dnest_norm_mean")
        if v is not None and np.isfinite(v):
            s.append(v)
    return s

# which cell for each family (STRATA optimal, matches head-to-head reported)
cells = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

out = {}
for fam, path in cells.items():
    rows = load(path)
    if rows is None:
        out[fam] = {"error": "no_file", "path": path}
        continue
    s = dnest_series(rows)
    if not s:
        out[fam] = {"error": "no_dnest", "n_rows": len(rows)}
        continue
    n = len(s)
    half = n // 2
    early = float(np.mean(s[:half]))
    late = float(np.mean(s[half:]))
    out[fam] = {
        "n_dnest_points": n, "dnest_early_1sthalf": early, "dnest_late_2ndhalf": late,
        "delta": late - early, "first5": [round(x, 4) for x in s[:5]],
        "last5": [round(x, 4) for x in s[-5:]], "path": path,
    }

print(json.dumps(out, indent=2))
