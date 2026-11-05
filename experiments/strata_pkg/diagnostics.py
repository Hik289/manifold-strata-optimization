"""诊断量 (README §5.3 全量日志 hook + §7 理论对应 + §10.12 synthetic test)。

提供:
- principal_angles: 两子空间主角
- nesting_gap: δ_nest = Σ_k ‖(I-P_{k+1})P_k‖_F^2
- reconstruction_error: tr[(I-P_k)C]
- commutator_proxy: Comm(i,j) = ‖C_i C_j - C_j C_i‖_F / (‖C_i‖_F ‖C_j‖_F + ε)
- top_r_subspace: 协方差 top-r 特征子空间
- flag_distance: d_Flag^2 = Σ_k α_k ‖P̂_k - P_k*‖_F^2
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def top_r_subspace(C: torch.Tensor, r: int) -> torch.Tensor:
    """对称半正定 C 的 top-r 特征向量 (n x r)。"""
    evals, evecs = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)
    return evecs[:, idx[:r]].contiguous()


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """两正交基 A (n x p), B (n x q) 张成子空间的主角 (弧度, 升序)。"""
    Qa, _ = torch.linalg.qr(A)
    Qb, _ = torch.linalg.qr(B)
    s = torch.linalg.svdvals(Qa.transpose(-1, -2) @ Qb)
    s = torch.clamp(s, -1.0, 1.0)
    return torch.arccos(s)


def projection_from_basis(U: torch.Tensor) -> torch.Tensor:
    return U @ U.transpose(-1, -2)


def nesting_gap(bases: Sequence[torch.Tensor]) -> float:
    """δ_nest = Σ_{k=1}^{K-1} ‖(I - P_{k+1}) P_k‖_F^2 (README §2.3 H1)。
    bases[k] 为第 k 层累计子空间正交基 (n x r_k), 需满足 r_1<...<r_K。
    """
    n = bases[0].shape[0]
    I = torch.eye(n, dtype=bases[0].dtype, device=bases[0].device)
    total = 0.0
    for k in range(len(bases) - 1):
        Pk = projection_from_basis(bases[k])
        Pk1 = projection_from_basis(bases[k + 1])
        gap = (I - Pk1) @ Pk
        total += float(gap.pow(2).sum())
    return total


def reconstruction_error(C: torch.Tensor, U: torch.Tensor) -> float:
    """tr[(I - P) C], P = U U^T (README §2.1 目标)。"""
    n = C.shape[0]
    I = torch.eye(n, dtype=C.dtype, device=C.device)
    P = projection_from_basis(U)
    return float(torch.trace((I - P) @ C))


def commutator_proxy(Ci: torch.Tensor, Cj: torch.Tensor, eps: float = 1e-12) -> float:
    """Comm(i,j) = ‖Ci Cj - Cj Ci‖_F / (‖Ci‖_F ‖Cj‖_F + ε) (README §7.1)。"""
    comm = Ci @ Cj - Cj @ Ci
    denom = torch.linalg.norm(Ci) * torch.linalg.norm(Cj) + eps
    return float(torch.linalg.norm(comm) / denom)


def flag_distance_sq(U_hat: torch.Tensor, U_star: torch.Tensor,
                     ranks: Sequence[int], alphas: Sequence[float] = None) -> float:
    """d_Flag^2 = Σ_k α_k ‖P̂_k - P_k*‖_F^2 (README §4.4 Theorem 10)。"""
    K = len(ranks)
    if alphas is None:
        alphas = [1.0] * K
    total = 0.0
    for k in range(K):
        rk = ranks[k]
        Ph = projection_from_basis(U_hat[:, :rk])
        Ps = projection_from_basis(U_star[:, :rk])
        total += alphas[k] * float((Ph - Ps).pow(2).sum())
    return total


def collect_flag_stats(state, ranks: Sequence[int]) -> dict:
    """从 optimizer state 抽取一次 flag update 的诊断 (§5.3 hook 7-10)。"""
    U = state["U"]
    bases = [U[:, :ranks[k]] for k in range(len(ranks))]
    return {
        "nesting_gap": nesting_gap(bases),
        "orthogonality": float((U.transpose(-1, -2) @ U -
                                torch.eye(U.shape[1], dtype=U.dtype, device=U.device)).pow(2).sum().sqrt()),
    }


# --------------------------------------------------------------------------- #
# implementation 要求的大尺度 δ_nest: 各时间尺度【独立最优子空间】之间的嵌套缺口
# (EDA H1 定义, 不是 STRATA 内部旗恒满足的 nesting_gap)
# --------------------------------------------------------------------------- #
def _ctilde_from_sketch(Y: torch.Tensor, Omega: torch.Tensor, gamma: float) -> torch.Tensor:
    """从 Nyström sketch 恢复 C̃_k = Y B^{-1} Y^T, B = sym(Ω^T Y) + γI (README §3.5)。"""
    B = 0.5 * (Omega.transpose(-1, -2) @ Y + (Omega.transpose(-1, -2) @ Y).transpose(-1, -2))
    B = B + gamma * torch.eye(B.shape[0], dtype=Y.dtype, device=Y.device)
    sol = torch.linalg.solve(B, Y.transpose(-1, -2))
    return Y @ sol


def independent_nesting_gap(Cs: Sequence[torch.Tensor], ranks: Sequence[int]):
    """EDA-δ_nest: 各时间尺度独立最优子空间 V̂_k=Top_{r_k}(C_k) 之间的嵌套缺口。

    δ_nest = Σ_{k=1}^{K-1} ‖(I − P̂_{k+1}) P̂_k‖²_F,  P̂_k = V̂_k V̂_k^T

    返回 (delta_nest, per_pair, ranks) —— 与"完美嵌套=0"和"随机零假设"对照。
    注意: 这里 V̂_k 是【各尺度独立 PCA】, 不强制嵌套, 所以 δ_nest>0 才有信息;
    δ_nest≈0 说明大尺度下各尺度独立最优子空间自发嵌套 (旗结构成立)。
    """
    K = len(ranks)
    n = Cs[0].shape[0]
    dev, dt = Cs[0].device, Cs[0].dtype
    I = torch.eye(n, dtype=dt, device=dev)
    Vs = [top_r_subspace(Cs[k], ranks[k]) for k in range(K)]
    per_pair = []
    total = 0.0
    for k in range(K - 1):
        Pk = Vs[k] @ Vs[k].transpose(-1, -2)
        Pk1 = Vs[k + 1] @ Vs[k + 1].transpose(-1, -2)
        g = float(((I - Pk1) @ Pk).pow(2).sum())
        per_pair.append(g)
        total += g
    return total, per_pair, list(ranks)


def random_nesting_null(ranks: Sequence[int], n: int, dtype, device,
                        n_samples: int = 20, seed: int = 0) -> float:
    """随机零假设: 独立随机 r_k 维子空间之间的 δ_nest 期望 (蒙特卡洛)。
    用于判定实测 δ_nest 是否显著低于"无结构"基线。
    """
    K = len(ranks)
    I = torch.eye(n, dtype=dtype, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    vals = []
    for _ in range(n_samples):
        Vs = []
        for k in range(K):
            Z = torch.randn(n, ranks[k], generator=g, dtype=dtype, device=device)
            Q, _ = torch.linalg.qr(Z)
            Vs.append(Q[:, :ranks[k]])
        tot = 0.0
        for k in range(K - 1):
            Pk = Vs[k] @ Vs[k].transpose(-1, -2)
            Pk1 = Vs[k + 1] @ Vs[k + 1].transpose(-1, -2)
            tot += float(((I - Pk1) @ Pk).pow(2).sum())
        vals.append(tot)
    return float(sum(vals) / len(vals))
