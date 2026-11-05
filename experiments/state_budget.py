"""Optimizer-state 标量预算统计 (README §3.12)。

用于 matched optimizer-state budget 对齐的量化说明。只统计 2D 矩阵参数上
低秩方法维护的额外状态; 1D/bias/norm 参数各方法都走标准 AdamW (相同), 单列。

November revision: counts all persistent tensor categories, including fixed
probes Omega and EMA sketches Y. The Independent baseline also caches an
orthonormal union basis (rank at most R). Formulas apply only to eligible
matrix parameters and omit shared full-Adam fallback parameters, Python
metadata, gradients, activations and temporary workspaces. Historical JSON
reports used a partial count excluding probes and are not rewritten.
"""

from __future__ import annotations

import torch.nn as nn


def _matrix_shapes(model, R):
    """列出走低秩分支的矩阵 (min>R) 的 (work_rows=max, d=min, numel)。
    conv-2D 化: 4D conv [out,in,kh,kw] 视作 2D [out, in*kh*kw] (implementation 约定)。
    """
    shapes = []
    for p in model.parameters():
        if p.dim() == 2:
            m, n = p.shape
        elif p.dim() == 4:
            m, n = p.shape[0], p.shape[1] * p.shape[2] * p.shape[3]
        else:
            continue
        if min(m, n) > R:
            shapes.append((max(m, n), min(m, n), m * n))
    return shapes


def budget_report(model, R, ranks, K, p, seed_note=""):
    """返回各方法 optimizer-state 标量总数 (仅 2D 矩阵分支) 的字典。"""
    shapes = _matrix_shapes(model, R)
    full_adamw = 0
    galore = 0
    single_g = 0
    indep_g = 0
    strata = 0
    # 独立块秩
    qs = [ranks[0]] + [ranks[k] - ranks[k - 1] for k in range(1, K)]
    sum_rk_plus_p = sum(ranks[k] + p for k in range(K))  # legacy nominal width; actual widths capped below
    for work_rows, d, mn in shapes:
        full_adamw += 2 * mn
        galore += 2 * work_rows * R + d * R
        single_g += 2 * work_rows * R + d * R + 2 * d * min(d, R + p)
        indep_g += 2 * work_rows * R + 2 * d * R + 2 * d * sum(min(d, q + p) for q in qs)
        strata += 2 * work_rows * R + d * R + 2 * d * sum(min(d, rk + p) for rk in ranks)
    return {
        "R": R, "ranks": list(ranks), "K": K, "p": p,
        "n_matrix_params": len(shapes),
        "state_scalars": {
            "full_adamw_2mn": full_adamw,
            "galore": galore,
            "single_grassmann": single_g,
            "independent_grassmann": indep_g,
            "strata": strata,
        },
        "ratio_vs_full": {
            "galore": galore / full_adamw if full_adamw else None,
            "single_grassmann": single_g / full_adamw if full_adamw else None,
            "independent_grassmann": indep_g / full_adamw if full_adamw else None,
            "strata": strata / full_adamw if full_adamw else None,
        },
        "note": seed_note + " November revision: includes fixed probes; independent includes a rank-R upper bound for union cache; eligible matrices only, not whole-model peak memory.",
    }
