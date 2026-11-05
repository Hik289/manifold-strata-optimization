"""STRATA §10 全部 12 组数值单元测试 (anchor_1) + §10.9/anchor_4 K=1 逐步等价。

每组 test 报告: 实测数值 (measured) vs 阈值 (threshold) vs 通过与否 (passed)。
全部用 float64 + CPU, 小 synthetic 张量, seed=42。

运行:
    python test_numerical.py            # 打印 + 写 experiments/unittest_report.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_pkg import geometry as geo  # noqa: E402
from strata_pkg import diagnostics as diag  # noqa: E402
from strata_pkg.strata_adamw import STRATAAdamW, _init_flag_basis, _orthonormal_sketch  # noqa: E402
from strata_pkg.baselines import SingleGrassmannAdamW  # noqa: E402

torch.set_default_dtype(torch.float64)
JST = timezone(timedelta(hours=9))

SEED = 42
RESULTS = []


def record(test_id, name, measured, threshold, passed, extra=None):
    entry = {
        "test": test_id,
        "name": name,
        "measured": measured,
        "threshold": threshold,
        "passed": bool(passed),
    }
    if extra:
        entry["detail"] = extra
    RESULTS.append(entry)
    status = "PASS" if passed else "FAIL"
    mstr = measured if isinstance(measured, str) else f"{measured:.3e}"
    tstr = threshold if isinstance(threshold, str) else f"{threshold:.1e}"
    print(f"[{status}] §{test_id} {name}: measured={mstr}  threshold={tstr}")
    return passed


def make_flag_basis(d, ranks, seed=SEED):
    """构造一个合法旗基 U ∈ St(d,R)。"""
    g = torch.Generator().manual_seed(seed)
    return _init_flag_basis(d, ranks[-1], g, torch.float64, torch.device("cpu"))


# ==========================================================================
# §10.1 正交性  ‖U^T U - I‖_F < 1e-5
# ==========================================================================
def test_10_1():
    d, ranks = 40, [8, 16, 24]
    U = make_flag_basis(d, ranks)
    R = ranks[-1]
    err = float((U.t() @ U - torch.eye(R)).norm())
    record("10.1", "orthogonality ‖UᵀU−I‖_F", err, 1e-5, err < 1e-5)


# ==========================================================================
# §10.2 嵌套性  ‖(I - P_{k+1}) P_k‖_F < 1e-5
# ==========================================================================
def test_10_2():
    d, ranks = 40, [8, 16, 24]
    U = make_flag_basis(d, ranks)
    n = d
    I = torch.eye(n)
    max_err = 0.0
    for k in range(1, len(ranks)):
        Pk = geo.cumulative_projection(U, ranks, k)
        Pk1 = geo.cumulative_projection(U, ranks, k + 1)
        err = float(((I - Pk1) @ Pk).norm())
        max_err = max(max_err, err)
    record("10.2", "nesting ‖(I−P_{k+1})P_k‖_F", max_err, 1e-5, max_err < 1e-5)


# ==========================================================================
# §10.3 分解正确性  ‖G - Σ G Π_k - G Π_⊥‖_F < 1e-6 ‖G‖_F
# ==========================================================================
def test_10_3():
    d, ranks = 40, [8, 16, 24]
    m = 30
    U = make_flag_basis(d, ranks)
    slices = geo.block_slices(ranks)
    g = torch.Generator().manual_seed(SEED + 1)
    G = torch.randn(m, d, generator=g)
    _, Gks, G_perp = geo.decompose_gradient(G, U, slices)
    recon = sum(Gks) + G_perp
    abs_err = float((G - recon).norm())
    rel = abs_err / float(G.norm())
    record("10.3", "decomposition rel err", rel, 1e-6, rel < 1e-6,
           extra={"abs_err": abs_err, "‖G‖_F": float(G.norm())})


# ==========================================================================
# §10.4 能量守恒  | ‖G‖² - Σ‖GΠ_k‖² - ‖GΠ_⊥‖² | < 1e-6 ‖G‖²
# ==========================================================================
def test_10_4():
    d, ranks = 40, [8, 16, 24]
    m = 30
    U = make_flag_basis(d, ranks)
    slices = geo.block_slices(ranks)
    g = torch.Generator().manual_seed(SEED + 2)
    G = torch.randn(m, d, generator=g)
    Zs, Gks, G_perp = geo.decompose_gradient(G, U, slices)
    total_energy = float(G.pow(2).sum())
    parts = sum(float(Gk.pow(2).sum()) for Gk in Gks) + float(G_perp.pow(2).sum())
    abs_err = abs(total_energy - parts)
    rel = abs_err / total_energy
    # 同时校验 §3.3 恒等式 ‖G‖² = Σ‖Z_k‖² + ‖G_⊥‖²
    zsum = sum(float(Z.pow(2).sum()) for Z in Zs) + float(G_perp.pow(2).sum())
    record("10.4", "energy conservation rel err", rel, 1e-6, rel < 1e-6,
           extra={"‖G‖²": total_energy, "Σ‖Z_k‖²+‖G⊥‖²": zsum,
                  "Z-identity_rel": abs(total_energy - zsum) / total_energy})


# ==========================================================================
# §10.5 Gauge invariance: U^[k] -> U^[k] O_k 不改变 P_j / 目标 / 重构梯度
# ==========================================================================
def test_10_5():
    d, ranks = 40, [8, 16, 24]
    m = 30
    U = make_flag_basis(d, ranks)
    slices = geo.block_slices(ranks)
    K = len(ranks)
    g = torch.Generator().manual_seed(SEED + 3)
    G = torch.randn(m, d, generator=g)

    # 各块右乘随机正交 O_k
    U2 = U.clone()
    for k in range(K):
        s = slices[k]
        q = s.stop - s.start
        A = torch.randn(q, q, generator=g)
        Ok, _ = torch.linalg.qr(A)
        U2[:, s] = U[:, s] @ Ok

    # (a) 累计投影 P_j 不变
    max_P = 0.0
    for j in range(1, K + 1):
        P1 = geo.cumulative_projection(U, ranks, j)
        P2 = geo.cumulative_projection(U2, ranks, j)
        max_P = max(max_P, float((P1 - P2).norm()))
    # (b) 旗目标 Φ 不变 (用 A_j = C 作占位)
    C = (G.t() @ G)
    A_list = [C for _ in range(K)]
    phi1 = float(geo.flag_objective(U, A_list, slices))
    phi2 = float(geo.flag_objective(U2, A_list, slices))
    # (c) 重构梯度 Σ G Π_k 不变
    _, Gk1, gp1 = geo.decompose_gradient(G, U, slices)
    _, Gk2, gp2 = geo.decompose_gradient(G, U2, slices)
    recon_diff = float((sum(Gk1) + gp1 - (sum(Gk2) + gp2)).norm())

    worst = max(max_P, abs(phi1 - phi2), recon_diff)
    record("10.5", "gauge invariance (max Δ P_j/Φ/recon)", worst, 1e-10, worst < 1e-10,
           extra={"ΔP_max": max_P, "ΔΦ": abs(phi1 - phi2), "Δrecon": recon_diff})


# ==========================================================================
# §10.6 水平投影: 水平梯度与垂直空间正交
# 垂直空间 = {U B : B 块对角反对称... } 实际检验 U^T H_hor 的块对角部分为 0
# 且 sym(U^T H_hor) = 0 (Stiefel 切空间条件)
# ==========================================================================
def test_10_6():
    d, ranks = 40, [8, 16, 24]
    U = make_flag_basis(d, ranks)
    slices = geo.block_slices(ranks)
    g = torch.Generator().manual_seed(SEED + 4)
    H = torch.randn(d, ranks[-1], generator=g)
    H_hor = geo.horizontal_gradient(U, H, slices)
    UtH = U.t() @ H_hor
    # Stiefel 条件: sym(U^T H_hor) = 0
    sym_err = float(geo.sym(UtH).norm())
    # 水平条件: 块对角部分 = 0 (无块内旋转分量)
    blk = geo.blkdiag_part(UtH, slices)
    blk_err = float(blk.norm())
    worst = max(sym_err, blk_err)
    record("10.6", "horizontal ⟂ vertical (sym & blkdiag)", worst, 1e-10, worst < 1e-10,
           extra={"‖sym(UᵀH_hor)‖": sym_err, "‖blkdiag(UᵀH_hor)‖": blk_err})


# ==========================================================================
# §10.7 Retraction: retraction 后 U_new^T U_new = I
# ==========================================================================
def test_10_7():
    d, ranks = 40, [8, 16, 24]
    U = make_flag_basis(d, ranks)
    slices = geo.block_slices(ranks)
    g = torch.Generator().manual_seed(SEED + 5)
    H = torch.randn(d, ranks[-1], generator=g)
    H_hor = geo.horizontal_gradient(U, H, slices)
    U_new = geo.qr_retraction(U, 1e-2 * H_hor)
    R = ranks[-1]
    err = float((U_new.t() @ U_new - torch.eye(R)).norm())
    record("10.7", "retraction orthonormality", err, 1e-5, err < 1e-5)


# ==========================================================================
# §10.8 Transport: U_new = U 时 transport 退化为恒等
# Procrustes(U, U) = I, 且 M -> M O^T = M
# ==========================================================================
def test_10_8():
    d, ranks = 40, [8, 16, 24]
    U = make_flag_basis(d, ranks)
    slices = geo.block_slices(ranks)
    K = len(ranks)
    max_err = 0.0
    for k in range(K):
        s = slices[k]
        O = geo.procrustes(U[:, s], U[:, s])
        q = s.stop - s.start
        err = float((O - torch.eye(q)).norm())
        max_err = max(max_err, err)
    record("10.8", "transport degenerates to identity", max_err, 1e-10, max_err < 1e-10)


# ==========================================================================
# §10.9 单层退化: K=1 STRATA == Single-Grassmann (逐步轨迹)
# 这是 anchor_4 的基础版; anchor_4 独立成 test_anchor4 报更严阈值
# ==========================================================================
def test_10_9():
    d, m, R = 40, 30, 12
    torch.manual_seed(SEED)
    W1 = torch.randn(m, d)
    W2 = W1.clone()
    W1.requires_grad_(True)
    W2.requires_grad_(True)

    common = dict(lr=1e-3, rho=0.999, xi=1e-3, tau=5, gamma=1e-6, p=8, eps=1e-8,
                  weight_decay=0.01, seed=SEED)
    opt_strata = STRATAAdamW([W1], lr=1e-3, ranks=[R], rhos=[0.999], beta1s=[0.9],
                             beta2s=[0.999], a_ks=[1.0], a_perp=1.0, xi=1e-3, tau=5,
                             gamma=1e-6, p=8, eps=1e-8, weight_decay=0.01, seed=SEED)
    opt_single = SingleGrassmannAdamW([W2], R=R, beta1=0.9, beta2=0.999, a=1.0,
                                      a_perp=1.0, **common)

    g = torch.Generator().manual_seed(SEED + 100)
    Wtar = torch.randn(m, d, generator=g)
    max_diff = 0.0
    for t in range(60):
        loss1 = ((W1 - Wtar) ** 2).sum()
        opt_strata.zero_grad(); loss1.backward(); opt_strata.step()
        loss2 = ((W2 - Wtar) ** 2).sum()
        opt_single.zero_grad(); loss2.backward(); opt_single.step()
        with torch.no_grad():
            diff = float((W1 - W2).abs().max())
        max_diff = max(max_diff, diff)
    record("10.9", "K=1 ≡ Single-Grassmann (max|ΔW|, 60 steps)", max_diff, 1e-6, max_diff < 1e-6)


# ==========================================================================
# §10.10 全秩退化: R=d 时 Π_⊥ = 0
# ==========================================================================
def test_10_10():
    d = 24
    ranks = [8, 16, 24]  # R = d
    U = make_flag_basis(d, ranks)
    Pi_perp = geo.residual_projection(U)
    err = float(Pi_perp.norm())
    record("10.10", "full-rank Π_⊥ = 0", err, 1e-10, err < 1e-10)


# ==========================================================================
# §10.11 有限差分: 旗目标 Riemannian 方向导数 vs 有限差分
# ==========================================================================
def test_10_11():
    d, ranks = 40, [8, 16, 24]
    U = make_flag_basis(d, ranks)
    slices = geo.block_slices(ranks)
    K = len(ranks)
    g = torch.Generator().manual_seed(SEED + 6)
    # 构造对称正定 A_j
    B = torch.randn(d, d, generator=g)
    C = B @ B.t() / d
    A_list = [C for _ in range(K)]

    # 环境梯度 -> 水平梯度 (Riemannian grad)
    H = geo.flag_ambient_gradient(U, A_list, slices)
    H_hor = geo.horizontal_gradient(U, H, slices)

    # 沿 H_hor 方向的解析方向导数 = <grad Φ, H_hor> = <H_hor, H_hor>
    # (因为 H_hor 是 Φ 的 Riemannian 梯度, ambient grad 内积在切空间上 = ‖grad‖²)
    analytic = float((H_hor * H_hor).sum())

    # 有限差分: (Φ(Retr(U, ε V)) - Φ(Retr(U, -ε V))) / 2ε, V = H_hor
    eps_fd = 1e-6
    Up = geo.qr_retraction(U, eps_fd * H_hor)
    Um = geo.qr_retraction(U, -eps_fd * H_hor)
    phi_p = float(geo.flag_objective(Up, A_list, slices))
    phi_m = float(geo.flag_objective(Um, A_list, slices))
    fd = (phi_p - phi_m) / (2 * eps_fd)

    rel = abs(analytic - fd) / (abs(analytic) + 1e-12)
    record("10.11", "finite-diff vs Riemannian deriv (rel)", rel, 1e-4, rel < 1e-4,
           extra={"analytic": analytic, "finite_diff": fd})


# ==========================================================================
# §10.12 synthetic 理论测试 (对照 Theorem 1/4/5/11)
# ==========================================================================
def test_10_12():
    detail = {}
    passed_all = True

    # ---- (a) 共享特征基下全局最优旗 = 按累计收益分块 (Theorem 1) ----
    d = 30
    ranks = [3, 6, 10]
    K = len(ranks)
    slices = geo.block_slices(ranks)
    omegas = [1.0, 1.0, 1.0]
    g = torch.Generator().manual_seed(SEED + 7)
    # 共同特征基 V, 各 C_k = V diag(λ_k) V^T, 交换
    A = torch.randn(d, d, generator=g)
    V, _ = torch.linalg.qr(A)
    # 构造 λ: 让 s_{j,i}=Σ_{k>=j} ω_k λ_{k,i} 的排序稳定, 内层核心方向收益最高
    base = torch.linspace(10, 1, d)  # 递减, 前几个方向能量最高
    lams = [base * (1.0 + 0.1 * k) for k in range(K)]
    Cs = [V @ torch.diag(lams[k]) @ V.t() for k in range(K)]

    # 理论最优旗: 按 s_{j,i} 分块, 内层取收益最高方向
    # 累计收益 s_{j,i} = Σ_{k>=j} ω_k λ_{k,i}; 对每个 i 相同排序 (因 λ 单调)
    # => 最优旗前 r_1 = top-3 方向, 依次嵌套
    U_star = V[:, :ranks[-1]].contiguous()
    # 数值优化: 从随机初始跑 Riemannian ascent, 应收敛到 span(V[:, :R]) 且分块正确
    U = make_flag_basis(d, ranks, seed=SEED + 8)
    for _ in range(4000):
        A_list = []
        for j in range(K):
            acc = torch.zeros(d, d)
            for k in range(j, K):
                acc = acc + omegas[k] * Cs[k]
            A_list.append(acc)
        H = geo.flag_ambient_gradient(U, A_list, slices)
        H_hor = geo.horizontal_gradient(U, H, slices)
        U = geo.qr_retraction(U, 5e-3 * H_hor)
    # 比较收敛旗与理论最优旗的每层累计子空间 (principal angle)
    max_angle = 0.0
    for k in range(K):
        ang = diag.principal_angles(U[:, :ranks[k]], U_star[:, :ranks[k]])
        max_angle = max(max_angle, float(ang.max()))
    detail["a_theorem1_max_principal_angle"] = max_angle
    a_pass = max_angle < 1e-2
    passed_all &= a_pass

    # ---- (b) 非最优临界点有负曲率方向 (strict saddle, Theorem 4) ----
    # 构造: 把低收益方向放内层 block1, 高收益方向放外层 block3 => 精确临界点(一阶=0)
    # 但按 Thm4 该点为 strict saddle: 沿"交换内/外层方向"的两维旋转测地线, Φ 有正二阶曲率
    # (对最大化 Φ 而言二阶正曲率 = 存在上升方向, 等价于 -Φ 的负曲率)。
    U_bad = V[:, :ranks[-1]].clone()
    perm = list(range(ranks[-1]))
    perm[0], perm[ranks[-1] - 1] = perm[ranks[-1] - 1], perm[0]  # 内层含低收益 v_{R-1}, 外层含高收益 v_0
    U_bad = U_bad[:, perm].contiguous()
    A_list = []
    for j in range(K):
        acc = torch.zeros(d, d)
        for k in range(j, K):
            acc = acc + omegas[k] * Cs[k]
        A_list.append(acc)

    # (b1) 确认是临界点: Riemannian 梯度范数 ≈ 0
    H = geo.flag_ambient_gradient(U_bad, A_list, slices)
    H_hor = geo.horizontal_gradient(U_bad, H, slices)
    grad_norm = float(H_hor.norm())

    # (b2) 二阶曲率: 沿内层列0 与外层列(R-1)的 2D 旋转测地线, 有限差分二阶导
    c0 = U_bad[:, 0].clone()
    cl = U_bad[:, ranks[-1] - 1].clone()

    def phi_along(alpha):
        Ua = U_bad.clone()
        Ua[:, 0] = math.cos(alpha) * c0 + math.sin(alpha) * cl
        Ua[:, ranks[-1] - 1] = -math.sin(alpha) * c0 + math.cos(alpha) * cl
        return float(geo.flag_objective(Ua, A_list, slices))

    h = 1e-3
    phi_p, phi_0, phi_m = phi_along(h), phi_along(0.0), phi_along(-h)
    curvature = (phi_p - 2 * phi_0 + phi_m) / (h * h)  # d²Φ/dα² at α=0
    detail["b_saddle_grad_norm"] = grad_norm
    detail["b_second_deriv_curvature"] = curvature
    # strict saddle: 临界(grad≈0) 且沿交换测地线正二阶曲率(可上升脱离)
    b_pass = (grad_norm < 1e-6) and (curvature > 1e-3)
    passed_all &= b_pass

    # ---- (c) 误差随 ε/Δ 缩放 (Theorem 2/5) ----
    # 近交换: C_k = C̄_k + ε E, 最优旗偏移 ~ O(ε/Δ)
    d2 = 20
    ranks2 = [4, 8]
    slices2 = geo.block_slices(ranks2)
    g2 = torch.Generator().manual_seed(SEED + 9)
    Vb, _ = torch.linalg.qr(torch.randn(d2, d2, generator=g2))
    lam_a = torch.linspace(10, 1, d2)
    Cbar = [Vb @ torch.diag(lam_a) @ Vb.t() for _ in range(2)]
    E = torch.randn(d2, d2, generator=g2); E = (E + E.t()) / 2; E = E / E.norm()

    def solve_flag(Cs_, ranks_, slices_, steps=3000, lr=5e-3):
        U_ = make_flag_basis(d2, ranks_, seed=SEED + 11)
        Kk = len(ranks_)
        for _ in range(steps):
            Al = []
            for j in range(Kk):
                acc = torch.zeros(d2, d2)
                for kk in range(j, Kk):
                    acc = acc + Cs_[kk]
                Al.append(acc)
            Hh = geo.flag_ambient_gradient(U_, Al, slices_)
            Hhor = geo.horizontal_gradient(U_, Hh, slices_)
            U_ = geo.qr_retraction(U_, lr * Hhor)
        return U_

    U_ref = solve_flag(Cbar, ranks2, slices2)
    offsets = []
    eps_list = [0.05, 0.1, 0.2, 0.4]
    for ev in eps_list:
        Cs_pert = [Cbar[k] + ev * E for k in range(2)]
        U_p = solve_flag(Cs_pert, ranks2, slices2)
        offsets.append(math.sqrt(diag.flag_distance_sq(U_p, U_ref, ranks2)))
    # 检验单调递增 (偏移随 ε 增大)
    mono = all(offsets[i] <= offsets[i + 1] + 1e-3 for i in range(len(offsets) - 1))
    detail["c_offsets_by_eps"] = {str(e): o for e, o in zip(eps_list, offsets)}
    c_pass = mono
    passed_all &= c_pass

    # ---- (d) tracking error 随 drift 增大 (Theorem 11) ----
    # 旋转真实旗, drift 越大, 在线单步跟踪残差越大
    d3 = 20
    ranks3 = [4, 8]
    slices3 = geo.block_slices(ranks3)
    g3 = torch.Generator().manual_seed(SEED + 13)
    Vc, _ = torch.linalg.qr(torch.randn(d3, d3, generator=g3))
    lam_c = torch.linspace(10, 1, d3)
    track_errs = []
    drift_list = [0.02, 0.05, 0.1, 0.2]
    for dr in drift_list:
        # 目标旗 = Vc 前 R 列; 施加旋转生成器强度 dr
        gen = torch.randn(d3, d3, generator=g3); gen = gen - gen.t()
        Rot = torch.linalg.matrix_exp(dr * gen / gen.norm())
        Vc2 = Rot @ Vc
        U_star_t = Vc2[:, :ranks3[-1]].contiguous()
        # 当前在线旗 = 上一时刻最优 (Vc), 单步 Riemannian 更新后测与新最优距离
        C_new = [Vc2 @ torch.diag(lam_c) @ Vc2.t() for _ in range(2)]
        U_cur = Vc[:, :ranks3[-1]].contiguous()
        Al = []
        for j in range(2):
            acc = torch.zeros(d3, d3)
            for kk in range(j, 2):
                acc = acc + C_new[kk]
            Al.append(acc)
        Hh = geo.flag_ambient_gradient(U_cur, Al, slices3)
        Hhor = geo.horizontal_gradient(U_cur, Hh, slices3)
        U_next = geo.qr_retraction(U_cur, 1e-3 * Hhor)
        te = math.sqrt(diag.flag_distance_sq(U_next, U_star_t, ranks3))
        track_errs.append(te)
    mono_d = all(track_errs[i] <= track_errs[i + 1] + 1e-3 for i in range(len(track_errs) - 1))
    detail["d_track_err_by_drift"] = {str(dd): t for dd, t in zip(drift_list, track_errs)}
    d_pass = mono_d
    passed_all &= d_pass

    measured_str = (f"T1_angle={max_angle:.2e}, saddle_curv={curvature:.2e}, "
                    f"eps_mono={mono}, drift_mono={mono_d}")
    record("10.12", "synthetic theory (T1 optimal flag / saddle / ε-scale / drift)",
           measured_str, "all 4 sub-tests pass", passed_all, extra=detail)


# ==========================================================================
# anchor_4 (§10.9 强化): K=1 STRATA 与 Single-Grassmann 逐步一致 < 1e-6
# 独立成一个 test, 多矩阵形状 + 更长步数
# ==========================================================================
def test_anchor4():
    shapes = [(30, 40), (50, 20), (40, 40)]  # 右旗/左旗/方阵
    R = 10
    max_diff_all = 0.0
    per_shape = {}
    for si, (m, n) in enumerate(shapes):
        torch.manual_seed(SEED + si)
        W1 = torch.randn(m, n)
        W2 = W1.clone()
        W1.requires_grad_(True); W2.requires_grad_(True)
        opt_s = STRATAAdamW([W1], lr=2e-3, ranks=[R], rhos=[0.995], beta1s=[0.9],
                            beta2s=[0.999], a_ks=[1.0], a_perp=1.0, xi=2e-3, tau=7,
                            gamma=1e-6, p=8, eps=1e-8, weight_decay=0.02, seed=SEED)
        opt_g = SingleGrassmannAdamW([W2], lr=2e-3, R=R, rho=0.995, beta1=0.9, beta2=0.999,
                                     a=1.0, a_perp=1.0, xi=2e-3, tau=7, gamma=1e-6, p=8,
                                     eps=1e-8, weight_decay=0.02, seed=SEED)
        g = torch.Generator().manual_seed(SEED + 200 + si)
        Wtar = torch.randn(m, n, generator=g)
        sh_max = 0.0
        for t in range(100):
            l1 = ((W1 - Wtar) ** 2).sum()
            opt_s.zero_grad(); l1.backward(); opt_s.step()
            l2 = ((W2 - Wtar) ** 2).sum()
            opt_g.zero_grad(); l2.backward(); opt_g.step()
            with torch.no_grad():
                sh_max = max(sh_max, float((W1 - W2).abs().max()))
        per_shape[f"{m}x{n}"] = sh_max
        max_diff_all = max(max_diff_all, sh_max)
    record("anchor_4", "K=1 STRATA ≡ Single-Grassmann trajectory (max|ΔW|, 100 steps, 3 shapes)",
           max_diff_all, 1e-6, max_diff_all < 1e-6, extra=per_shape)


# ==========================================================================
def main():
    tests = [test_10_1, test_10_2, test_10_3, test_10_4, test_10_5, test_10_6,
             test_10_7, test_10_8, test_10_9, test_10_10, test_10_11, test_10_12,
             test_anchor4]
    print("=" * 70)
    print("STRATA §10 数值单元测试 (anchor_1) + anchor_4")
    print(f"torch {torch.__version__}, dtype {torch.get_default_dtype()}, seed {SEED}")
    print("=" * 70)
    for tfn in tests:
        try:
            tfn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            RESULTS.append({"test": tfn.__name__, "name": tfn.__name__,
                            "measured": "EXCEPTION", "threshold": "-",
                            "passed": False, "error": str(e)})
            print(f"[FAIL] {tfn.__name__}: EXCEPTION {e}")

    n_pass = sum(1 for r in RESULTS if r["passed"])
    n_total = len(RESULTS)
    print("=" * 70)
    print(f"总计: {n_pass}/{n_total} 通过")
    print("=" * 70)

    report = {
        "project_id": "manifoldstrata",
        "phase": "IMPL+UNITTEST",
        "anchor": ["H0.anchor_1", "H0.anchor_4"],
        "timestamp": datetime.now(JST).isoformat(),
        "torch_version": torch.__version__,
        "dtype": str(torch.get_default_dtype()),
        "seed": SEED,
        "n_passed": n_pass,
        "n_total": n_total,
        "all_passed": n_pass == n_total,
        "tests": RESULTS,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unittest_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告写入: {out}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
