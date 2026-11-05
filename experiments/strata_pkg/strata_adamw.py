"""STRATA-AdamW 主优化器 (README §3, Algorithm 1)。

作为标准 PyTorch optimizer 使用。二维权重矩阵走旗流形分支;
一维/bias/norm 参数走标准 AdamW 分支 (README §3.4 末)。

实现约定见 geometry.py。左/右侧旗通过内部 transpose 统一 (README §3.9)。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.optim.optimizer import Optimizer

from . import geometry as geo


def _as_matrix(p_tensor):
    """把参数视作 2D 矩阵用于旗分支 (implementation conv-2D 化约定):
    - 2D [m,n]  -> 原样, orig_shape=(m,n)
    - 4D conv [out,in,kh,kw] -> [out, in*kh*kw], orig_shape=原4D
    - 其他 (1D/bias/norm) -> (None, None) 走标准 AdamW
    返回 (mat2d_or_None, orig_shape)。
    """
    if p_tensor.dim() == 2:
        return p_tensor, tuple(p_tensor.shape)
    if p_tensor.dim() == 4:
        out = p_tensor.shape[0]
        return p_tensor.reshape(out, -1), tuple(p_tensor.shape)
    return None, None


class _ShapeProxy:
    """给 _init_matrix_state 提供 2D shape + 原 tensor 的 dtype/device。"""
    def __init__(self, mat2d, orig_tensor):
        self.shape = tuple(mat2d.shape)
        self.dtype = orig_tensor.dtype
        self.device = orig_tensor.device


def _orthonormal_sketch(d: int, s: int, generator: torch.Generator, dtype, device) -> torch.Tensor:
    """采样固定正交 sketch Ω ∈ R^{d x s}, Ω^T Ω = I_s (README §3.2)。"""
    Z = torch.randn(d, s, generator=generator, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(Z, mode="reduced")
    dsign = torch.sign(torch.diagonal(R))
    dsign = torch.where(dsign == 0, torch.ones_like(dsign), dsign)
    Q = Q * dsign.unsqueeze(0)
    return Q[:, :s]


def _init_flag_basis(d: int, R: int, generator: torch.Generator, dtype, device) -> torch.Tensor:
    """初始化旗基 U_0 = qf(Z), Z~N(0,I)^{d x R} (README §3.2)。"""
    Z = torch.randn(d, R, generator=generator, dtype=dtype, device=device)
    Q, Rm = torch.linalg.qr(Z, mode="reduced")
    dsign = torch.sign(torch.diagonal(Rm))
    dsign = torch.where(dsign == 0, torch.ones_like(dsign), dsign)
    Q = Q * dsign.unsqueeze(0)
    return Q[:, :R]


class STRATAAdamW(Optimizer):
    """STRATA-AdamW。

    每个二维矩阵参数维护旗状态 S = (U, {M_k,V_k,c_k}, {Y_k})。
    超参数在 group 级别配置; 一维参数走标准 AdamW。

    group 关键字段:
        lr, weight_decay, eps
        ranks:  [r_1,...,r_K] 累计秩 (r_K = R = max rank)
        rhos:   [ρ_1,...,ρ_K] 从长到短时间尺度
        beta1s, beta2s: 各块 Adam 常数 (长度 K)
        a_ks:   各块更新系数 (长度 K), a_perp: 残差系数
        omegas: 旗目标权重 ω_k (长度 K, 默认全 1)
        xi:     旗学习率
        tau:    旗更新间隔
        gamma:  Nyström 正则项
        p:      sketch oversampling (s_k = r_k + p)
        seed:   每矩阵状态初始化种子基
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        ranks: Sequence[int] = (16, 32, 64),
        rhos: Sequence[float] = (0.999, 0.99, 0.9),
        beta1s: Optional[Sequence[float]] = None,
        beta2s: Optional[Sequence[float]] = None,
        a_ks: Optional[Sequence[float]] = None,
        a_perp: float = 1.0,
        omegas: Optional[Sequence[float]] = None,
        xi: float = 1e-3,
        tau: int = 25,
        gamma: float = 1e-6,
        p: int = 8,
        eps: float = 1e-6,  # spec 修正(EXP-013): 旗投影使 flag-block Z_k 可真near-zero→V_k≈0,
                            # eps=1e-8 时 H=M/√Vhat 爆炸(2e5)。1e-6 floor 数值稳定。见 EXPERIMENT_LOG。
        weight_decay: float = 0.0,
        seed: int = 42,
        # 一维参数的标准 AdamW 常数
        adam_betas=(0.9, 0.999),
    ):
        ranks = list(ranks)
        K = len(ranks)
        if beta1s is None:
            beta1s = [0.9] * K
        if beta2s is None:
            beta2s = [0.999] * K
        if a_ks is None:
            a_ks = [1.0] * K
        if omegas is None:
            omegas = [1.0] * K
        assert len(beta1s) == K and len(beta2s) == K and len(a_ks) == K and len(rhos) == K and len(omegas) == K
        assert all(0 < ranks[i] < ranks[i + 1] for i in range(K - 1)) or K == 1
        defaults = dict(
            lr=lr, ranks=ranks, rhos=list(rhos), beta1s=list(beta1s), beta2s=list(beta2s),
            a_ks=list(a_ks), a_perp=a_perp, omegas=list(omegas), xi=xi, tau=tau,
            gamma=gamma, p=p, eps=eps, weight_decay=weight_decay, seed=seed, adam_betas=adam_betas,
        )
        super().__init__(params, defaults)
        self._param_counter = 0  # 用于给每个矩阵参数分配独立种子

    # ------------------------------------------------------------------ #
    def _init_matrix_state(self, state, p_tensor, group):
        m, n = p_tensor.shape
        transposed = n > m
        work_rows, d = (n, m) if transposed else (m, n)  # 旗维 d = min(m,n)
        ranks = group["ranks"]
        R = ranks[-1]
        K = len(ranks)
        p = group["p"]
        dtype, device = p_tensor.dtype, p_tensor.device

        seed_base = group["seed"] * 100003 + self._param_counter
        self._param_counter += 1
        gU = torch.Generator(device=device).manual_seed(seed_base)
        gO = torch.Generator(device=device).manual_seed(seed_base + 1)

        state["transposed"] = transposed
        state["d"] = d
        state["work_rows"] = work_rows
        state["slices"] = geo.block_slices(ranks)
        state["U"] = _init_flag_basis(d, R, gU, dtype, device)
        state["M"] = [torch.zeros(work_rows, ranks[k] - (ranks[k - 1] if k > 0 else 0),
                                  dtype=dtype, device=device) for k in range(K)]
        state["V"] = [torch.zeros_like(state["M"][k]) for k in range(K)]
        state["c_M"] = [0 for _ in range(K)]   # M 的 bias-correction 计数 (永不重置)
        state["c_V"] = [0 for _ in range(K)]   # V 的 bias-correction 计数 (每 flag update 重置)
        s_list = [min(d, ranks[k] + p) for k in range(K)]
        state["Omega"] = [_orthonormal_sketch(d, s_list[k], gO, dtype, device) for k in range(K)]
        state["Y"] = [torch.zeros(d, s_list[k], dtype=dtype, device=device) for k in range(K)]
        state["t"] = 0
        state["last_stats"] = {}

    # ------------------------------------------------------------------ #
    def _matrix_update(self, state, G, group):
        """返回参数空间的更新方向 D_real (与 p_tensor 同形)。"""
        transposed = state["transposed"]
        work = G.t().contiguous() if transposed else G
        U = state["U"]
        slices = state["slices"]
        ranks = group["ranks"]
        K = len(ranks)
        rhos, b1s, b2s, a_ks = group["rhos"], group["beta1s"], group["beta2s"], group["a_ks"]
        omegas = group["omegas"]
        eps = group["eps"]
        a_perp = group["a_perp"]
        mw, d = work.shape

        # 1. 多时间尺度 Nyström sketch 更新 (README §3.5)
        for k in range(K):
            Omega_k = state["Omega"][k]
            Gw_Omega = work @ Omega_k            # (mw x s_k)
            C_Omega = work.t() @ Gw_Omega        # (d x s_k) = C @ Ω_k, C = work^T work
            state["Y"][k] = rhos[k] * state["Y"][k] + (1.0 - rhos[k]) * C_Omega

        # 2. 层级低维 AdamW (README §3.4)
        # 修复 (RUNNING-wave1 DEBUG): 分离 M/V 的 bias-correction counter。
        # M 在 flag update 时被 transport 保留 (M←M·Oᵀ, 成熟动量), 其 bias correction
        # 必须用"自初始化以来的总步数" c_M (永不重置); V 在 flag update 时重置为 0,
        # 其 bias correction 用"自上次重置以来步数" c_V。原实现共用一个重置的 c → M 被
        # 过校正 (reset 后一步 c=1 → Mhat=10·M), 每 τ 步踢一次 → grad_norm 累积爆炸。
        D = torch.zeros_like(work)
        for k in range(K):
            Ublk = U[:, slices[k]]
            Z = work @ Ublk                      # (mw x q_k)
            state["M"][k] = b1s[k] * state["M"][k] + (1.0 - b1s[k]) * Z
            state["V"][k] = b2s[k] * state["V"][k] + (1.0 - b2s[k]) * (Z * Z)
            state["c_M"][k] += 1
            state["c_V"][k] += 1
            Mhat = state["M"][k] / (1.0 - b1s[k] ** state["c_M"][k])
            Vhat = state["V"][k] / (1.0 - b2s[k] ** state["c_V"][k])
            H = Mhat / (torch.sqrt(Vhat) + eps)
            D = D + a_ks[k] * (H @ Ublk.t())

        # 3. 无状态残差方向 (README §3.4)
        # G_perp = work Π_⊥ = work - Σ_k (work U^[k]) U^[k]^T = work - work U U^T
        G_perp = work - (work @ U) @ U.t()
        denom = torch.linalg.norm(G_perp) / (mw * d) ** 0.5 + eps
        D = D + a_perp * (G_perp / denom)

        # 记录层级能量 (§5.3 hook)
        with torch.no_grad():
            energies = [float((work @ U[:, slices[k]]).pow(2).sum()) for k in range(K)]
            state["last_stats"]["level_energy"] = energies
            state["last_stats"]["residual_energy"] = float(G_perp.pow(2).sum())

        # 4. 旗更新 (README §3.6-3.8), 每 τ 步
        if state["t"] % group["tau"] == 0:
            self._flag_update(state, work, group)

        state["t"] += 1
        return D.t().contiguous() if transposed else D

    # ------------------------------------------------------------------ #
    def _flag_update(self, state, work, group):
        U = state["U"]
        slices = state["slices"]
        ranks = group["ranks"]
        K = len(ranks)
        omegas = group["omegas"]
        gamma = group["gamma"]
        xi = group["xi"]

        # 计算 Ctilde_k @ U (不构造 d x d): (d x R)
        CU = []
        for k in range(K):
            CU.append(geo.nystrom_apply(state["Y"][k], state["Omega"][k], U, gamma))

        # 环境梯度 H = [2 A_j U^[j]], A_j = Σ_{k>=j} ω_k Ctilde_k
        H_cols = []
        for j in range(K):
            acc = torch.zeros(state["d"], slices[j].stop - slices[j].start, dtype=U.dtype, device=U.device)
            for k in range(j, K):
                acc = acc + omegas[k] * CU[k][:, slices[j]]
            H_cols.append(2.0 * acc)
        H = torch.cat(H_cols, dim=1)

        H_hor = geo.horizontal_gradient(U, H, slices)
        U_bar = geo.qr_retraction(U, xi * H_hor)

        # Procrustes 块对齐 + 状态处理 (README §3.8)
        new_U = torch.empty_like(U)
        for k in range(K):
            s = slices[k]
            O_k = geo.procrustes(U_bar[:, s], U[:, s])
            new_U[:, s] = U_bar[:, s] @ O_k
            state["M"][k] = state["M"][k] @ O_k.t()            # transport 保留 (c_M 不重置)
            state["V"][k] = torch.zeros_like(state["V"][k])   # 二阶矩重置
            state["c_V"][k] = 0                                # 仅 V 的 counter 重置
        state["U"] = new_U

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            b1, b2 = group["adam_betas"]
            for p_tensor in group["params"]:
                if p_tensor.grad is None:
                    continue
                grad = p_tensor.grad
                state = self.state[p_tensor]

                # conv-2D 化 (implementation 约束): 4D conv [out,in,kh,kw] reshape 成
                # 2D [out, in*kh*kw] 走旗分支; 2D 直接走。
                mat2d, orig_shape = _as_matrix(p_tensor)
                use_flag = (mat2d is not None) and (min(mat2d.shape) > group["ranks"][-1])

                if use_flag:
                    # ---- 旗流形分支 ----
                    if "U" not in state:
                        self._init_matrix_state(state, _ShapeProxy(mat2d, p_tensor), group)
                    grad2d = grad.reshape(orig_shape[0], -1) if p_tensor.dim() == 4 else grad
                    D = self._matrix_update(state, grad2d, group)
                    if p_tensor.dim() == 4:
                        D = D.reshape(orig_shape)
                    p_tensor.mul_(1.0 - lr * wd)
                    p_tensor.add_(D, alpha=-lr)
                else:
                    # ---- 标准 AdamW 分支 (1D/bias/norm 或 min<=R) ----
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p_tensor)
                        state["exp_avg_sq"] = torch.zeros_like(p_tensor)
                        state["step_adam"] = 0
                    state["step_adam"] += 1
                    m_, v_ = state["exp_avg"], state["exp_avg_sq"]
                    m_.mul_(b1).add_(grad, alpha=1 - b1)
                    v_.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    bc1 = 1 - b1 ** state["step_adam"]
                    bc2 = 1 - b2 ** state["step_adam"]
                    denom = (v_ / bc2).sqrt().add_(eps)
                    p_tensor.mul_(1.0 - lr * wd)
                    p_tensor.addcdiv_(m_ / bc1, denom, value=-lr)
        return loss
