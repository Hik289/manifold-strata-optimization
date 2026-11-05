"""旗流形几何原语 (README §3.1-3.8, §3.5-3.6)。

约定 (右侧旗, n <= m):
- U ∈ St(n, R), 按块拼接 U = [U^[1], ..., U^[K]], U^[k] ∈ R^{n x q_k}, U^T U = I_R
- P_k  = U_{1:k} U_{1:k}^T                      (第 k 层累计投影, n x n)
- Π_k  = U^[k] U^[k]^T                          (第 k 层增量投影)
- Π_⊥  = I - P_K = I - U U^T                    (残差投影)
- 梯度 G ∈ R^{m x n}, 右乘投影: G Π_k, G Π_⊥

所有函数默认 float64 输入以保证数值测试阈值。左侧旗 (m<n) 由调用方对 G 转置后复用本模块。
"""

from __future__ import annotations

from typing import List, Sequence

import torch


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def sym(X: torch.Tensor) -> torch.Tensor:
    """对称化 (X + X^T)/2。"""
    return 0.5 * (X + X.transpose(-1, -2))


def block_slices(ranks: Sequence[int]) -> List[slice]:
    """由累计秩 [r_1,...,r_K] (r_0=0 隐含) 生成各块在 [0,R) 上的列 slice。"""
    slices = []
    prev = 0
    for r in ranks:
        slices.append(slice(prev, r))
        prev = r
    return slices


def blkdiag_part(M: torch.Tensor, slices: Sequence[slice]) -> torch.Tensor:
    """取 R x R 矩阵 M 的块对角部分 (各 q_k x q_k 对角块), 其余置零。"""
    out = torch.zeros_like(M)
    for s in slices:
        out[s, s] = M[s, s]
    return out


# --------------------------------------------------------------------------- #
# 投影 (README §2.2, §3.1, §3.3)
# --------------------------------------------------------------------------- #
def cumulative_projection(U: torch.Tensor, ranks: Sequence[int], k: int) -> torch.Tensor:
    """P_k = U_{1:r_k} U_{1:r_k}^T。k 为 1-based 层号。"""
    rk = ranks[k - 1]
    Uk = U[:, :rk]
    return Uk @ Uk.transpose(-1, -2)


def incremental_projection(U: torch.Tensor, slices: Sequence[slice], k: int) -> torch.Tensor:
    """Π_k = U^[k] U^[k]^T。k 为 1-based 层号。"""
    Ublk = U[:, slices[k - 1]]
    return Ublk @ Ublk.transpose(-1, -2)


def residual_projection(U: torch.Tensor) -> torch.Tensor:
    """Π_⊥ = I - U U^T。"""
    n = U.shape[0]
    return torch.eye(n, dtype=U.dtype, device=U.device) - U @ U.transpose(-1, -2)


def decompose_gradient(G: torch.Tensor, U: torch.Tensor, slices: Sequence[slice]):
    """右侧旗梯度分解 (README §3.3):
        Z_k = G U^[k]                    (m x q_k, 低维坐标)
        G_k = Z_k U^[k]^T = G Π_k        (m x n)
        G_⊥ = G - Σ_k G_k = G Π_⊥
    返回 (list[Z_k], list[G_k], G_perp)。
    """
    Zs, Gks = [], []
    for s in slices:
        Ublk = U[:, s]
        Z = G @ Ublk
        Zs.append(Z)
        Gks.append(Z @ Ublk.transpose(-1, -2))
    G_perp = G - sum(Gks)
    return Zs, Gks, G_perp


# --------------------------------------------------------------------------- #
# 旗目标与水平 Riemannian 梯度 (README §3.6, §3.7)
# --------------------------------------------------------------------------- #
def flag_objective(U: torch.Tensor, A_list: Sequence[torch.Tensor], slices: Sequence[slice]) -> torch.Tensor:
    """Φ(U) = Σ_j tr[ U^[j]^T A_j U^[j] ]  (README §3.6 展开形式)。
    A_j = Σ_{k>=j} ω_k Ctilde_k, 由调用方预先累加好后传入 A_list (1-based 对应 j=1..K)。
    """
    val = U.new_zeros(())
    for j, s in enumerate(slices):
        Ublk = U[:, s]
        val = val + torch.trace(Ublk.transpose(-1, -2) @ A_list[j] @ Ublk)
    return val


def flag_ambient_gradient(U: torch.Tensor, A_list: Sequence[torch.Tensor], slices: Sequence[slice]) -> torch.Tensor:
    """环境梯度 H = [2 A_1 U^[1], ..., 2 A_K U^[K]] (README §3.6)。"""
    cols = []
    for j, s in enumerate(slices):
        cols.append(2.0 * A_list[j] @ U[:, s])
    return torch.cat(cols, dim=1)


def horizontal_gradient(U: torch.Tensor, H: torch.Tensor, slices: Sequence[slice]) -> torch.Tensor:
    """旗流形水平梯度 (README §3.7):
        H_st  = H - U sym(U^T H)                (投影到 Stiefel 切空间)
        H_hor = H_st - U blkdiag(U^T H_st)      (去除块内旋转的垂直分量)
    """
    UtH = U.transpose(-1, -2) @ H
    H_st = H - U @ sym(UtH)
    UtH_st = U.transpose(-1, -2) @ H_st
    H_hor = H_st - U @ blkdiag_part(UtH_st, slices)
    return H_hor


# --------------------------------------------------------------------------- #
# QR retraction + Procrustes 块对齐 (README §3.8)
# --------------------------------------------------------------------------- #
def qr_retraction(U: torch.Tensor, Delta: torch.Tensor) -> torch.Tensor:
    """QR retraction: qf(U + Δ), 符号通过强制 R 对角线为正固定。"""
    Q, R = torch.linalg.qr(U + Delta, mode="reduced")
    d = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    Q = Q * d.unsqueeze(-2)
    return Q[:, : U.shape[1]]


def procrustes(U_bar_blk: torch.Tensor, U_blk: torch.Tensor) -> torch.Tensor:
    """块 Procrustes 对齐 (README §3.8):
        L Σ R^T = U_bar^[k]^T U^[k],   O_k = L R^T   (q_k x q_k 正交)
    使 U_bar^[k] O_k 在 Frobenius 意义下最接近 U^[k]。
    """
    M = U_bar_blk.transpose(-1, -2) @ U_blk
    # 鲁棒性 (发散后 NaN/inf 保护): 输入非有限时返回单位阵, 让 run 优雅记录为发散态
    # 而非崩溃整个 driver。不改变有效路径行为。
    q = M.shape[-1]
    if not torch.isfinite(M).all():
        return torch.eye(q, dtype=M.dtype, device=M.device)
    try:
        L, _, Rt = torch.linalg.svd(M)
    except Exception:
        return torch.eye(q, dtype=M.dtype, device=M.device)
    return L @ Rt


# --------------------------------------------------------------------------- #
# 多时间尺度 Nyström 统计 (README §3.5)
# --------------------------------------------------------------------------- #
def nystrom_apply(Y: torch.Tensor, Omega: torch.Tensor, X: torch.Tensor, gamma: float) -> torch.Tensor:
    """计算 Ctilde_k X = Y_k B_k^{-1} (Y_k^T X), 其中 B_k = sym(Ω_k^T Y_k) + γ I。
    不构造 n x n 矩阵。
    """
    B = sym(Omega.transpose(-1, -2) @ Y)
    B = B + gamma * torch.eye(B.shape[0], dtype=Y.dtype, device=Y.device)
    YtX = Y.transpose(-1, -2) @ X
    try:
        sol = torch.linalg.solve(B, YtX)
    except Exception:
        # 数值兜底: 仅当 B 奇异时 (通常因上游发散使 Y 含 NaN/Inf); 健康路径不触发,
        # 故不影响 anchor_4 逐位等价。lstsq 对奇异/病态更稳。
        sol = torch.linalg.lstsq(B, YtX).solution
    return Y @ sol


def nystrom_matrix(Y: torch.Tensor, Omega: torch.Tensor, gamma: float) -> torch.Tensor:
    """显式构造 Ctilde_k = Y B^{-1} Y^T (仅用于测试/诊断, 训练不调用)。"""
    n = Y.shape[0]
    I = torch.eye(n, dtype=Y.dtype, device=Y.device)
    return nystrom_apply(Y, Omega, I, gamma)
