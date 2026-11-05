"""Baseline optimizers: GaLore / Single-Grassmann / Independent-Grassmann。

设计目标 (README §3.12 状态预算): 与 STRATA 在 matched max rank R + matched
optimizer-state budget 下公平对比。

- SingleGrassmannAdamW: 单一子空间 U ∈ Gr(R,d), 低维 AdamW 状态, 在线 Riemannian
  子空间更新 (等价于 STRATA K=1)。独立代码路径, 用于 anchor_4 逐步等价校验。
- GaLore: 官方语义 —— 梯度低秩投影 G U, 低维 Adam, 周期 SVD 刷新投影 (Zhao et al. 2024)。
- IndependentGrassmannAdamW: K 个独立子空间 (无嵌套约束), 总秩 = R, 各自 Adam 状态。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.optim.optimizer import Optimizer

from . import geometry as geo
from .strata_adamw import _init_flag_basis, _orthonormal_sketch, _as_matrix, _ShapeProxy


# =========================================================================== #
# Single-Grassmann (STRATA K=1 的配对独立实现) —— anchor_4 校验对象
# =========================================================================== #
class SingleGrassmannAdamW(Optimizer):
    """单一 Grassmann 子空间 AdamW。

    与 K=1 STRATA 完全对应 (README §10.9)。为保证 anchor_4 逐步等价:
    - init RNG 逻辑与 STRATA._init_matrix_state 一致 (同 seed_base 公式)
    - 更新算术与 STRATA K=1 分支逐字对应
    """

    def __init__(self, params, lr=1e-3, R=64, rho=0.999, beta1=0.9, beta2=0.999,
                 a=1.0, a_perp=1.0, xi=1e-3, tau=25, gamma=1e-6, p=8, eps=1e-6,
                 weight_decay=0.0, seed=42, adam_betas=(0.9, 0.999)):
        defaults = dict(lr=lr, R=R, rho=rho, beta1=beta1, beta2=beta2, a=a, a_perp=a_perp,
                        xi=xi, tau=tau, gamma=gamma, p=p, eps=eps, weight_decay=weight_decay,
                        seed=seed, adam_betas=adam_betas)
        super().__init__(params, defaults)
        self._param_counter = 0

    def _init_state(self, state, p_tensor, group):
        m, n = p_tensor.shape
        transposed = n > m
        work_rows, d = (n, m) if transposed else (m, n)
        R, p = group["R"], group["p"]
        dtype, device = p_tensor.dtype, p_tensor.device
        seed_base = group["seed"] * 100003 + self._param_counter
        self._param_counter += 1
        gU = torch.Generator(device=device).manual_seed(seed_base)
        gO = torch.Generator(device=device).manual_seed(seed_base + 1)
        state["transposed"] = transposed
        state["d"] = d
        state["U"] = _init_flag_basis(d, R, gU, dtype, device)
        state["M"] = torch.zeros(work_rows, R, dtype=dtype, device=device)
        state["V"] = torch.zeros(work_rows, R, dtype=dtype, device=device)
        state["c_M"] = 0   # M bias-correction counter (transport 保留, 不重置)
        state["c_V"] = 0   # V bias-correction counter (每 subspace update 重置)
        s = min(d, R + p)
        state["Omega"] = _orthonormal_sketch(d, s, gO, dtype, device)
        state["Y"] = torch.zeros(d, s, dtype=dtype, device=device)
        state["t"] = 0

    def _update(self, state, G, group):
        transposed = state["transposed"]
        work = G.t().contiguous() if transposed else G
        U = state["U"]
        rho, b1, b2 = group["rho"], group["beta1"], group["beta2"]
        eps, a, a_perp = group["eps"], group["a"], group["a_perp"]
        mw, d = work.shape

        # Nyström sketch
        Gw_Omega = work @ state["Omega"]
        C_Omega = work.t() @ Gw_Omega
        state["Y"] = rho * state["Y"] + (1.0 - rho) * C_Omega

        # 低维 AdamW
        Z = work @ U
        state["M"] = b1 * state["M"] + (1.0 - b1) * Z
        state["V"] = b2 * state["V"] + (1.0 - b2) * (Z * Z)
        state["c_M"] += 1
        state["c_V"] += 1
        Mhat = state["M"] / (1.0 - b1 ** state["c_M"])
        Vhat = state["V"] / (1.0 - b2 ** state["c_V"])
        H = Mhat / (torch.sqrt(Vhat) + eps)
        D = a * (H @ U.t())

        # 残差
        G_perp = work - (work @ U) @ U.t()
        denom = torch.linalg.norm(G_perp) / (mw * d) ** 0.5 + eps
        D = D + a_perp * (G_perp / denom)

        # 子空间更新 (单块 = 整个 Grassmann)
        if state["t"] % group["tau"] == 0:
            slices = [slice(0, group["R"])]
            CU = geo.nystrom_apply(state["Y"], state["Omega"], U, group["gamma"])
            Hflag = 2.0 * CU
            H_hor = geo.horizontal_gradient(U, Hflag, slices)
            U_bar = geo.qr_retraction(U, group["xi"] * H_hor)
            O = geo.procrustes(U_bar, U)
            state["U"] = U_bar @ O
            state["M"] = state["M"] @ O.t()
            state["V"] = torch.zeros_like(state["V"])
            state["c_V"] = 0   # 仅 V counter 重置; c_M 保留

        state["t"] += 1
        return D.t().contiguous() if transposed else D

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            b1, b2 = group["adam_betas"]
            for p_tensor in group["params"]:
                if p_tensor.grad is None:
                    continue
                grad = p_tensor.grad
                state = self.state[p_tensor]
                mat2d, orig_shape = _as_matrix(p_tensor)
                if (mat2d is not None) and (min(mat2d.shape) > group["R"]):
                    if "U" not in state:
                        self._init_state(state, _ShapeProxy(mat2d, p_tensor), group)
                    grad2d = grad.reshape(orig_shape[0], -1) if p_tensor.dim() == 4 else grad
                    D = self._update(state, grad2d, group)
                    if p_tensor.dim() == 4:
                        D = D.reshape(orig_shape)
                    p_tensor.mul_(1.0 - lr * wd)
                    p_tensor.add_(D, alpha=-lr)
                else:
                    _std_adamw(state, p_tensor, grad, lr, wd, eps, b1, b2)
        return loss


# =========================================================================== #
# GaLore (官方语义: 低秩投影 + 周期 SVD 刷新)
# =========================================================================== #
class GaLoreAdamW(Optimizer):
    """GaLore (Zhao et al., ICML 2024) 官方语义。

    每 tau 步用梯度 SVD 刷新投影 P (取 top-R 左/右奇异向量);
    低维空间 R_t = P^T G, 在 R_t 上跑 Adam, 更新投回全空间 P @ update。
    scale 系数按官方默认。
    """

    def __init__(self, params, lr=1e-3, R=64, tau=200, scale=1.0, beta1=0.9,
                 beta2=0.999, eps=1e-6, weight_decay=0.0, adam_betas=(0.9, 0.999)):
        defaults = dict(lr=lr, R=R, tau=tau, scale=scale, beta1=beta1, beta2=beta2,
                        eps=eps, weight_decay=weight_decay, adam_betas=adam_betas)
        super().__init__(params, defaults)

    def _get_proj(self, G, R):
        # 官方 GaLore: 在较小一侧投影。返回 (P, side) side='right' 用 G@P, 'left' 用 P^T@G
        m, n = G.shape
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        if m >= n:
            # 右投影 P = V[:, :R] (n x R), 低维 = G @ P (m x R)
            return Vh[:R, :].t().contiguous(), "right"
        else:
            # 左投影 P = U[:, :R] (m x R), 低维 = P^T @ G (R x n)
            return U[:, :R].contiguous(), "left"

    def _update(self, state, G, group):
        R, tau = group["R"], group["tau"]
        b1, b2, eps = group["beta1"], group["beta2"], group["eps"]
        if state["t"] % tau == 0:
            state["P"], state["side"] = self._get_proj(G, R)
        P, side = state["P"], state["side"]
        low = (G @ P) if side == "right" else (P.t() @ G)
        if "M" not in state or state["M"].shape != low.shape:
            state["M"] = torch.zeros_like(low)
            state["V"] = torch.zeros_like(low)
        state["M"] = b1 * state["M"] + (1 - b1) * low
        state["V"] = b2 * state["V"] + (1 - b2) * (low * low)
        state["c"] += 1
        c = state["c"]
        Mhat = state["M"] / (1 - b1 ** c)
        Vhat = state["V"] / (1 - b2 ** c)
        Hlow = Mhat / (torch.sqrt(Vhat) + eps)
        D = (Hlow @ P.t()) if side == "right" else (P @ Hlow)
        state["t"] += 1
        return group["scale"] * D

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            b1, b2 = group["adam_betas"]
            for p_tensor in group["params"]:
                if p_tensor.grad is None:
                    continue
                grad = p_tensor.grad
                state = self.state[p_tensor]
                mat2d, orig_shape = _as_matrix(p_tensor)
                if (mat2d is not None) and (min(mat2d.shape) > group["R"]):
                    if "t" not in state:
                        state["t"] = 0
                        state["c"] = 0
                    grad2d = grad.reshape(orig_shape[0], -1) if p_tensor.dim() == 4 else grad
                    D = self._update(state, grad2d, group)
                    if p_tensor.dim() == 4:
                        D = D.reshape(orig_shape)
                    p_tensor.mul_(1.0 - lr * wd)
                    p_tensor.add_(D, alpha=-lr)
                else:
                    _std_adamw(state, p_tensor, grad, lr, wd, eps, b1, b2)
        return loss


# =========================================================================== #
# Independent-Grassmann (K 个独立子空间, 无嵌套约束, 总秩 = R)
# =========================================================================== #
class IndependentGrassmannAdamW(Optimizer):
    """K 个互相独立的 Grassmann 子空间, 各秩 q_k, Σ q_k = R (matched 总秩预算)。

    与 STRATA 的区别: 不强制 U_1 ⊂ U_2 ⊂ ... 嵌套 (无块拼接的正交约束),
    每个子空间独立在线更新自身协方差方向。用于 A2 消融的正确性前提。
    """

    def __init__(self, params, lr=1e-3, ranks=(16, 32, 64), rhos=(0.999, 0.99, 0.9),
                 beta1s=None, beta2s=None, a_ks=None, a_perp=1.0, xi=1e-3, tau=25,
                 gamma=1e-6, p=8, eps=1e-6, weight_decay=0.0, seed=42, adam_betas=(0.9, 0.999)):
        cum = list(ranks)
        K = len(cum)
        qs = [cum[0]] + [cum[k] - cum[k - 1] for k in range(1, K)]  # 各独立块秩
        if beta1s is None:
            beta1s = [0.9] * K
        if beta2s is None:
            beta2s = [0.999] * K
        if a_ks is None:
            a_ks = [1.0] * K
        defaults = dict(lr=lr, qs=qs, R=cum[-1], rhos=list(rhos), beta1s=list(beta1s),
                        beta2s=list(beta2s), a_ks=list(a_ks), a_perp=a_perp, xi=xi, tau=tau,
                        gamma=gamma, p=p, eps=eps, weight_decay=weight_decay, seed=seed,
                        adam_betas=adam_betas)
        super().__init__(params, defaults)
        self._param_counter = 0

    def _init_state(self, state, p_tensor, group):
        m, n = p_tensor.shape
        transposed = n > m
        work_rows, d = (n, m) if transposed else (m, n)
        qs, p = group["qs"], group["p"]
        K = len(qs)
        dtype, device = p_tensor.dtype, p_tensor.device
        seed_base = group["seed"] * 100003 + self._param_counter
        self._param_counter += 1
        state["transposed"] = transposed
        state["d"] = d
        state["U"] = []
        state["Omega"] = []
        state["Y"] = []
        state["M"] = []
        state["V"] = []
        state["c_M"] = [0] * K
        state["c_V"] = [0] * K
        for k in range(K):
            gU = torch.Generator(device=device).manual_seed(seed_base + 10 * k)
            gO = torch.Generator(device=device).manual_seed(seed_base + 10 * k + 1)
            state["U"].append(_init_flag_basis(d, qs[k], gU, dtype, device))
            s = min(d, qs[k] + p)
            state["Omega"].append(_orthonormal_sketch(d, s, gO, dtype, device))
            state["Y"].append(torch.zeros(d, s, dtype=dtype, device=device))
            state["M"].append(torch.zeros(work_rows, qs[k], dtype=dtype, device=device))
            state["V"].append(torch.zeros(work_rows, qs[k], dtype=dtype, device=device))
        state["t"] = 0

    def _update(self, state, G, group):
        transposed = state["transposed"]
        work = G.t().contiguous() if transposed else G
        qs = group["qs"]
        K = len(qs)
        if "U_union" not in state:
            union, singular, _ = torch.linalg.svd(torch.cat(state["U"], dim=1), full_matrices=False)
            tol = max(work.shape[1], sum(qs)) * torch.finfo(work.dtype).eps * singular.max()
            state["U_union"] = union[:, singular > tol]
        U_res = state["U_union"]
        rhos, b1s, b2s, a_ks = group["rhos"], group["beta1s"], group["beta2s"], group["a_ks"]
        eps, a_perp = group["eps"], group["a_perp"]
        mw, d = work.shape
        D = torch.zeros_like(work)
        for k in range(K):
            Uk = state["U"][k]
            Gw_Omega = work @ state["Omega"][k]
            C_Omega = work.t() @ Gw_Omega
            state["Y"][k] = rhos[k] * state["Y"][k] + (1 - rhos[k]) * C_Omega
            Z = work @ Uk
            state["M"][k] = b1s[k] * state["M"][k] + (1 - b1s[k]) * Z
            state["V"][k] = b2s[k] * state["V"][k] + (1 - b2s[k]) * (Z * Z)
            state["c_M"][k] += 1
            state["c_V"][k] += 1
            Mhat = state["M"][k] / (1 - b1s[k] ** state["c_M"][k])
            Vhat = state["V"][k] / (1 - b2s[k] ** state["c_V"][k])
            Hk = Mhat / (torch.sqrt(Vhat) + eps)
            D = D + a_ks[k] * (Hk @ Uk.t())
            if state["t"] % group["tau"] == 0:
                slices = [slice(0, qs[k])]
                CU = geo.nystrom_apply(state["Y"][k], state["Omega"][k], Uk, group["gamma"])
                H_hor = geo.horizontal_gradient(Uk, 2.0 * CU, slices)
                U_bar = geo.qr_retraction(Uk, group["xi"] * H_hor)
                O = geo.procrustes(U_bar, Uk)
                state["U"][k] = U_bar @ O
                state["M"][k] = state["M"][k] @ O.t()
                state["V"][k] = torch.zeros_like(state["V"][k])
                state["c_V"][k] = 0   # 仅 V counter 重置
        # Residual must use the true union projector: independent bases overlap.
        # U_res is the old union basis, matching the block coordinates above.
        G_perp = work - (work @ U_res) @ U_res.t()
        D = D + a_perp * G_perp / (torch.linalg.norm(G_perp) / (mw * d) ** 0.5 + eps)
        if state["t"] % group["tau"] == 0:
            union, singular, _ = torch.linalg.svd(torch.cat(state["U"], dim=1), full_matrices=False)
            tol = max(d, sum(qs)) * torch.finfo(work.dtype).eps * singular.max()
            state["U_union"] = union[:, singular > tol]
        state["t"] += 1
        return D.t().contiguous() if transposed else D

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            b1, b2 = group["adam_betas"]
            for p_tensor in group["params"]:
                if p_tensor.grad is None:
                    continue
                grad = p_tensor.grad
                state = self.state[p_tensor]
                mat2d, orig_shape = _as_matrix(p_tensor)
                if (mat2d is not None) and (min(mat2d.shape) > group["R"]):
                    if "U" not in state:
                        self._init_state(state, _ShapeProxy(mat2d, p_tensor), group)
                    grad2d = grad.reshape(orig_shape[0], -1) if p_tensor.dim() == 4 else grad
                    D = self._update(state, grad2d, group)
                    if p_tensor.dim() == 4:
                        D = D.reshape(orig_shape)
                    p_tensor.mul_(1.0 - lr * wd)
                    p_tensor.add_(D, alpha=-lr)
                else:
                    _std_adamw(state, p_tensor, grad, lr, wd, eps, b1, b2)
        return loss


# =========================================================================== #
# 共享: 标准 AdamW (1D/bias/norm 分支)
# =========================================================================== #
def _std_adamw(state, p_tensor, grad, lr, wd, eps, b1, b2):
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
