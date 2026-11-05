"""
二次独立核验: δ_nest EPOCH 轴重判 (Data Scientist 核验闸)
从原始 log.jsonl 独立复算六族 δ_nest 的 epoch 轴趋势, 多种定义交叉验证:
 - Δ (early-half → late-half)
 - Δ (early-quartile Q1 → late-quartile Q4)   [implementation 指定]
 - 线性回归斜率 (dnest vs 归一化 epoch)
 - Spearman 秩相关 (单调性强度)
 - 起点值 (检验 'graph 唯一低起点 0.05')
 - 单调性: 是否严格/近似单调 (检验 'graph 唯一强单调坍缩')
不信任 dnest_cross_family.json, 直接从各族 STRATA 最优 cell 的 raw log 读 flag_diag。
在源机器分别跑 (PINN@training host, MLP/CNN/graph@各自机器)。
"""
import json, os, sys, math
import numpy as np

def load(path):
    if not os.path.exists(path): return None
    rows = []
    for l in open(path):
        l = l.strip()
        if not l: continue
        try: rows.append(json.loads(l))
        except: pass
    return rows or None

def dnest_epoch_series(rows):
    """返回 [(epoch, dnest_norm)] 按 epoch 升序 (跳过 warmup=第0个flag点前的null)。"""
    s = []
    for r in rows:
        fd = r.get("flag_diag") or {}
        v = fd.get("indep_dnest_norm_mean")
        if v is not None and np.isfinite(v):
            s.append((r["epoch"], float(v)))
    s.sort(key=lambda x: x[0])
    return s

def spearman(x, y):
    """无 scipy 时的 Spearman 秩相关。"""
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a)); r[order] = np.arange(len(a))
        return r
    rx, ry = rank(np.array(x)), rank(np.array(y))
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/denom) if denom > 0 else float("nan")

def analyze(name, path, metric):
    rows = load(path)
    if rows is None:
        return {"name": name, "error": "no_file", "path": path}
    s = dnest_epoch_series(rows)
    if len(s) < 8:
        return {"name": name, "error": f"too_few_dnest_pts={len(s)}", "path": path}
    eps = [p[0] for p in s]; dn = [p[1] for p in s]
    n = len(dn)
    q = n // 4
    # half/half
    half = n // 2
    d_half = float(np.mean(dn[half:]) - np.mean(dn[:half]))
    # quartile Q1 -> Q4
    q1 = float(np.mean(dn[:q])); q4 = float(np.mean(dn[-q:]))
    d_quart = q4 - q1
    # 线性斜率 (归一化 epoch 到 [0,1])
    xe = np.array(eps, dtype=float); xe = (xe - xe.min())/(xe.max()-xe.min()+1e-12)
    slope = float(np.polyfit(xe, dn, 1)[0])
    # Spearman
    rho = spearman(xe.tolist(), dn)
    # 起点 (前 5% 均值, 更稳健于单点)
    k0 = max(1, n//20)
    start = float(np.mean(dn[:k0]))
    end = float(np.mean(dn[-k0:]))
    return {
        "name": name, "n_dnest_pts": n, "metric": metric,
        "start_first5pct": round(start, 4), "end_last5pct": round(end, 4),
        "raw_first": round(dn[0], 4), "raw_last": round(dn[-1], 4),
        "delta_half": round(d_half, 4),
        "delta_quartile_Q1toQ4": round(d_quart, 4),
        "Q1_mean": round(q1, 4), "Q4_mean": round(q4, 4),
        "lin_slope_norm_epoch": round(slope, 4),
        "spearman_rho": round(rho, 4),
        "min": round(min(dn), 4), "max": round(max(dn), 4),
        "path": path,
    }

if __name__ == "__main__":
    # cells passed as JSON: {name: [path, metric]}
    cells = json.loads(sys.argv[1])
    out = {}
    for name, (path, metric) in cells.items():
        out[name] = analyze(name, path, metric)
    print(json.dumps(out, indent=2))
