# STRATA: Multi-Scale Manifold Optimization with Nested Gradient Subspaces

## 0. 项目定位

现代低秩优化器通常假设：一个参数矩阵在某一训练阶段只需要一个固定秩或周期更新的梯度子空间。GaLore 将全参数训练中的梯度投影到一个低秩子空间，从而减少优化器状态；GaLore 2 进一步处理大规模预训练中的投影更新、并行训练和系统代价。其他矩阵优化器，如 Shampoo、SOAP 和 Muon，则从矩阵预条件或正交化更新的角度改善训练效率。它们共同说明，梯度矩阵具有可利用的结构，但通常使用**单一空间、单一时间尺度或单一更新规则**。

另一方面，Grassmann 流形用于表示固定维数子空间，而旗流形表示一组嵌套子空间：

$$
\mathcal U_1
\subset
\mathcal U_2
\subset
\cdots
\subset
\mathcal U_K.
$$

已有旗流形研究给出了商流形表示、切空间、梯度、Hessian、指数映射和优化算法，也有工作使用旗结构保证不同维数的子空间估计保持嵌套一致。在线旗流形学习也已用于跟踪随时间变化的嵌套系统子空间。然而，这些研究并未把旗流形作为深度学习优化器状态，用于分离梯度中的长期、阶段性和快速变化成分。

本项目提出一个单一算法：

> **STRATA：在旗流形上在线学习嵌套梯度子空间，并在不同层级上使用不同的记忆时间、学习率和预条件状态。**

STRATA 不约束模型权重，不改变前向传播，也不学习权重奇异值。它只改变优化器如何表示和处理梯度。

对于一个梯度向量 $g_t\in\mathbb R^d$，STRATA 维护：

$$
\mathcal U_{1,t}
\subset
\mathcal U_{2,t}
\subset
\cdots
\subset
\mathcal U_{K,t}
\subset
\mathbb R^d,
$$

其中：

$$
\dim(\mathcal U_{k,t})=r_k,
\qquad
0<r_1<\cdots<r_K<d.
$$

梯度被分解为互相正交的层级成分：

$$
g_t
=
\sum_{k=1}^{K}
\Pi_{k,t}g_t
+
\Pi_{\perp,t}g_t,
$$

其中：

$$
\Pi_{1,t}=P_{1,t},
$$

$$
\Pi_{k,t}=P_{k,t}-P_{k-1,t},
\qquad k\ge2,
$$

$$
\Pi_{\perp,t}=I-P_{K,t}.
$$

不同层级分别表示：

- **稳定层**：长期重复出现、低漂移的方向；
- **过渡层**：随训练阶段变化、但在局部时间窗口内稳定的方向；
- **快速层**：短期有效、变化较快的方向；
- **残差层**：当前旗结构尚未解释的新方向和噪声方向。

核心科学问题不是“低秩是否有效”，而是：

> **深度学习梯度是否具有嵌套、多时间尺度的子空间结构；如果有，旗流形是否是比单一 Grassmann 子空间更合适的优化器状态空间？**

本项目所有实验只运行 **1 个 seed**。这是固定研究协议，不在主结果中报告多 seed 平均。所有长训练必须至少运行 **1000 epochs**，或在语言模型中运行与 1000 epochs 等价的完整长训练预算。每个 epoch、每次评估、每个 flag update 和每个 checkpoint 的结果都必须永久保存，用于后续绘制完整训练曲线、子空间轨迹和误差曲线。

---

# 1. 与现有研究的关系

## 1.1 低秩梯度优化

GaLore 在训练完整权重的同时，将梯度投影到低秩空间，并在低维空间中维护优化器状态。其关键假设是梯度存在缓慢变化的低秩结构。GaLore 2 进一步展示了低秩梯度投影可扩展到更长 token 预算和更大模型。

这类方法的典型形式是：

$$
\widetilde G_t
=
P_t^\top G_t
$$

或：

$$
\widetilde G_t
=
P_t^\top G_tQ_t,
$$

然后在单一低秩空间中维护一阶和二阶状态。

限制在于：

1. 一个投影空间需要同时覆盖长期和短期方向；
2. 所有低秩坐标通常共享同一动量时间常数；
3. 子空间更新常通过周期性 SVD 完成；
4. 子空间维数改变时，不保证不同维数解之间嵌套一致；
5. 单一秩 $r$ 不能表达“稳定核心 + 阶段扩展 + 快速外层”。

STRATA 不替代 GaLore 的内存目标，而是提出更强的结构假设：梯度并非只有一个低秩空间，而是具有嵌套层级。

## 1.2 Grassmann 子空间优化

Grassmann 流形：

$$
\operatorname{Gr}(r,d)
$$

表示 $\mathbb R^d$ 中所有 $r$ 维子空间。它天然适合在线 PCA、子空间跟踪和低秩表示。

若只维护：

$$
\mathcal U_t\in\operatorname{Gr}(r,d),
$$

则优化器只能回答“当前最重要的 $r$ 维空间是什么”，不能同时回答：

- 哪些方向在长时间内稳定；
- 哪些方向只在当前阶段重要；
- 哪些方向刚刚出现；
- 不同时间尺度分别需要多少维数。

## 1.3 旗流形优化

旗流形：

$$
\operatorname{Flag}(r_1,\ldots,r_K;d)
$$

表示固定维数序列的嵌套子空间。已有数学研究已经建立了旗流形上的商空间表示、Riemannian gradient、Hessian、geodesic 和数值优化工具。近期嵌套子空间学习工作进一步说明，将多个 Grassmann 问题提升到旗流形可保证不同维数解的一致嵌套性。

STRATA 与这些工作的区别是：

1. 优化对象不是数据表示，而是优化器中的梯度统计；
2. 旗状态随神经网络训练在线变化；
3. 每个旗层级有不同的时间尺度、动量和预条件；
4. 旗结构直接决定参数更新方向；
5. 理论需要同时分析流形状态误差和参数优化误差。

## 1.4 矩阵预条件优化器

Shampoo 使用 Kronecker 结构估计矩阵梯度的二阶统计。SOAP 将 Shampoo 的特征基与 Adam 型更新结合。Muon 对二维矩阵更新进行正交化或近似极分解处理。

这些方法主要改变：

$$
G_t
\mapsto
B_tG_tC_t.
$$

STRATA 主要改变：

$$
G_t
\mapsto
\sum_k
\eta_{k,t}
\Pi_{k,t}
\mathcal A_k(G_t)
+
\eta_{\perp,t}
\Pi_{\perp,t}
\mathcal A_\perp(G_t),
$$

其中 $\mathcal A_k$ 是每个层级自己的动量或自适应更新。STRATA 可以接在 SGD 或 AdamW 上，但旗层级结构本身是方法核心。

## 1.5 本项目的唯一新假设

本项目只检验以下假设：

> 梯度统计由共享稳定核心和逐层扩展的快速方向组成，因此不同时间尺度的最优子空间近似嵌套。

若该假设不成立，独立 Grassmann 子空间或单一低秩子空间可能更合适。实验必须保留这种失败情况。

---

# 2. 问题定义

## 2.1 多时间尺度梯度统计

给定随机梯度：

$$
g_t=\nabla_\theta\ell(\theta_t;\xi_t),
$$

定义 $K$ 个指数平滑协方差：

$$
C_{k,t}
=
\rho_k C_{k,t-1}
+
(1-\rho_k)g_tg_t^\top,
$$

其中：

$$
1>\rho_1>\rho_2>\cdots>\rho_K\ge0.
$$

$\rho_1$ 对应长时间尺度，$\rho_K$ 对应短时间尺度。

理想情况下，存在嵌套主子空间：

$$
\mathcal U_{1,t}^\star
\subset
\cdots
\subset
\mathcal U_{K,t}^\star
$$

使：

$$
\mathcal U_{k,t}^\star
\in
\arg\min_{\dim(\mathcal U)=r_k}
\operatorname{tr}
\left[
(I-P_{\mathcal U})C_{k,t}
\right].
$$

一般情况下，各 $C_{k,t}$ 的主子空间未必严格嵌套。因此 STRATA 优化联合目标：

$$
\mathcal F_t(\mathcal U_1,\ldots,\mathcal U_K)
=
\sum_{k=1}^{K}
\omega_k
\operatorname{tr}
\left[
(I-P_k)C_{k,t}
\right],
$$

subject to：

$$
\mathcal U_1\subset\cdots\subset\mathcal U_K.
$$

## 2.2 梯度层级

定义：

$$
P_{k,t}
=
\operatorname{Proj}_{\mathcal U_{k,t}},
$$

$$
P_{0,t}=0.
$$

增量层级投影为：

$$
\Pi_{k,t}
=
P_{k,t}-P_{k-1,t}.
$$

最大空间外的残差投影为：

$$
\Pi_{\perp,t}
=
I-P_{K,t}.
$$

更新算子：

$$
B_t
=
\sum_{k=1}^{K}
\eta_{k,t}\Pi_{k,t}
+
\eta_{\perp,t}\Pi_{\perp,t}.
$$

最简单的 STRATA-SGD 更新为：

$$
\theta_{t+1}
=
\theta_t-B_tm_t,
$$

其中 $m_t$ 由层级动量构成。

## 2.3 科学假设

### H1：嵌套近似成立

定义独立最优子空间：

$$
\widehat{\mathcal V}_{k,t}
=
\operatorname{Top}_{r_k}(C_{k,t}).
$$

嵌套缺口：

$$
\delta_{\mathrm{nest},t}
=
\sum_{k=1}^{K-1}
\|
(I-\widehat P_{k+1,t})
\widehat P_{k,t}
\|_F^2.
$$

若：

$$
\delta_{\mathrm{nest},t}
$$

长期较小，则旗结构合理。

### H2：不同层级具有不同漂移速度

定义：

$$
d_{k,t}
=
\|P_{k,t}-P_{k,t-1}\|_F.
$$

预期：

$$
\mathbb E[d_{1,t}]
<
\mathbb E[d_{2,t}]
<
\cdots
<
\mathbb E[d_{K,t}].
$$

### H3：层级更新优于统一更新

在固定最大秩和相同优化器状态预算下：

$$
\mathcal L_{\mathrm{STRATA}}
<
\mathcal L_{\mathrm{single}}
$$

或达到相同训练损失所需的步数更少。

### H4：优势来自嵌套结构

独立子空间、随机嵌套旗、固定旗和单一 Grassmann 不能同时复现：

- 主任务性能；
- 多时间尺度重构误差；
- 任务切换恢复；
- 状态稳定性。

---

# 3. STRATA 算法

本节只定义论文和代码中的唯一主算法：**STRATA-AdamW**。算法对每个二维权重矩阵维护一个单边旗流形状态，在每个旗增量块中维护低维 AdamW 状态，并用固定的多时间尺度 Nyström sketch 更新旗。除偏置、归一化参数和一维参数外，不再使用其他更新分支。

## 3.1 输入、状态与固定约定

考虑二维参数矩阵：

$$
W_t\in\mathbb R^{m\times n},
\qquad
G_t=\nabla_W\ell_t(W_t).
$$

STRATA 总是在较小的一侧维护旗。定义：

$$
d=\min(m,n),
\qquad
0=r_0<r_1<\cdots<r_K=R<d,
$$

$$
q_k=r_k-r_{k-1}.
$$

若 $n\le m$，使用右侧旗；若 $m<n$，使用左侧旗。下面先给出右侧旗的完整算法。左侧旗由转置得到，不再定义第二套算法。

右侧旗基为：

$$
U_t=
[U_t^{[1]},\ldots,U_t^{[K]}]
\in\operatorname{St}(n,R),
$$

其中：

$$
U_t^{[k]}\in\mathbb R^{n\times q_k},
\qquad
U_t^\top U_t=I_R.
$$

第 $k$ 个嵌套子空间及其增量投影为：

$$
\mathcal U_{k,t}
=
\operatorname{span}(U_t^{[1]},\ldots,U_t^{[k]}),
$$

$$
P_{k,t}=U_{1:k,t}U_{1:k,t}^\top,
\qquad
\Pi_{k,t}=U_t^{[k]}U_t^{[k]\top},
$$

$$
P_{0,t}=0,
\qquad
\Pi_{\perp,t}=I-P_{K,t}.
$$

每个矩阵参数只维护以下状态：

$$
\mathsf S_t=
\left
(U_t,
\{M_{k,t},V_{k,t}\}_{k=1}^{K},
\{Y_{k,t}\}_{k=1}^{K}
\right).
$$

其中：

$$
M_{k,t},V_{k,t}\in\mathbb R^{m\times q_k}
$$

是第 $k$ 个增量块的低维 Adam 状态，且：

$$
Y_{k,t}\in\mathbb R^{n\times s_k},
\qquad
s_k=r_k+p,
$$

是时间尺度 $k$ 的 Nyström sketch。最大旗空间外不维护一阶或二阶状态。

固定超参数为：

$$
\{r_k,\rho_k,\beta_{1,k},\beta_{2,k},a_k\}_{k=1}^{K},
\quad
\eta_t,
\quad
\lambda_{\mathrm{wd}},
\quad
\epsilon,
\quad
\xi,
\quad
\tau,
\quad
p,
\quad
a_\perp.
$$

其中：

$$
1>\rho_1>\rho_2>\cdots>\rho_K\ge0
$$

定义从长期到短期的统计时间尺度，$\tau$ 是固定旗更新间隔。算法运行期间不改变 $K$、$r_k$、sketch 维数或更新规则。

## 3.2 初始化

对每个矩阵参数，采样高斯矩阵并进行 QR 分解：

$$
Q_0R_0=\operatorname{qr}(Z_0),
\qquad
Z_0\sim\mathcal N(0,I)^{n\times R},
$$

$$
U_0=Q_0[:,1:R].
$$

所有低维优化器状态初始化为零：

$$
M_{k,0}=0,
\qquad
V_{k,0}=0.
$$

为每个时间尺度固定采样一次 sketch 矩阵：

$$
\Omega_k\in\mathbb R^{n\times s_k},
\qquad
\Omega_k^\top\Omega_k=I_{s_k},
$$

并初始化：

$$
Y_{k,0}=0.
$$

$\Omega_k$ 在整个训练过程中保持不变。

## 3.3 每步梯度坐标与正交分解

对右侧旗，第 $k$ 个增量块的低维梯度坐标为：

$$
Z_{k,t}=G_tU_t^{[k]}
\in\mathbb R^{m\times q_k}.
$$

对应的矩阵空间梯度分量为：

$$
G_{k,t}=Z_{k,t}U_t^{[k]\top}
=G_t\Pi_{k,t}.
$$

残差分量为：

$$
G_{\perp,t}
=G_t\Pi_{\perp,t}
=G_t-\sum_{k=1}^{K}G_{k,t}.
$$

由于 $\Pi_{1,t},\ldots,\Pi_{K,t},\Pi_{\perp,t}$ 两两正交：

$$
\langle G_{i,t},G_{j,t}\rangle_F=0
\quad(i\ne j),
$$

$$
\|G_t\|_F^2
=
\sum_{k=1}^{K}\|Z_{k,t}\|_F^2
+
\|G_{\perp,t}\|_F^2.
$$

## 3.4 层级低维 AdamW 更新

每个增量块使用独立的一阶和二阶时间常数：

$$
M_{k,t}
=
\beta_{1,k}M_{k,t-1}
+
(1-\beta_{1,k})Z_{k,t},
$$

$$
V_{k,t}
=
\beta_{2,k}V_{k,t-1}
+
(1-\beta_{2,k})Z_{k,t}^{\odot2}.
$$

令该状态自上次重置后已经连续更新 $c_{k,t}$ 次。偏差修正为：

$$
\widehat M_{k,t}
=
\frac{M_{k,t}}{1-\beta_{1,k}^{c_{k,t}}},
\qquad
\widehat V_{k,t}
=
\frac{V_{k,t}}{1-\beta_{2,k}^{c_{k,t}}}.
$$

第 $k$ 个块的低维预条件方向为：

$$
H_{k,t}
=
\frac{\widehat M_{k,t}}
{\sqrt{\widehat V_{k,t}}+\epsilon}.
$$

重构到参数空间：

$$
D_{k,t}=H_{k,t}U_t^{[k]\top}.
$$

最大旗空间外采用无状态残差方向：

$$
D_{\perp,t}
=
\frac{G_{\perp,t}}
{\|G_{\perp,t}\|_F/\sqrt{mn}+\epsilon}.
$$

因此，残差方向保留新出现的梯度信息，但不会引入 $m\times(n-R)$ 大小的 Adam 状态。

参数更新唯一地定义为：

$$
\boxed{
W_{t+1}
=
(1-\eta_t\lambda_{\mathrm{wd}})W_t
-
\eta_t
\left(
\sum_{k=1}^{K}a_kD_{k,t}
+
a_\perp D_{\perp,t}
\right).
}
$$

一维参数、偏置和归一化参数使用标准 AdamW，不进入 STRATA 旗状态。

## 3.5 多时间尺度 Nyström 统计

右侧梯度协方差观测为：

$$
C_t=G_t^\top G_t\in\mathbb R^{n\times n}.
$$

STRATA 不显式存储 $C_t$ 或其指数移动平均。对每个时间尺度 $k$，只更新：

$$
Y_{k,t}
=
\rho_kY_{k,t-1}
+
(1-\rho_k)C_t\Omega_k.
$$

该更新可按以下次序计算而不形成 $C_t$：

$$
C_t\Omega_k
=
G_t^\top(G_t\Omega_k).
$$

定义小矩阵：

$$
B_{k,t}
=
\operatorname{sym}(\Omega_k^\top Y_{k,t})
+
\gamma I_{s_k},
$$

其中 $\gamma>0$ 是固定数值正则项。时间尺度 $k$ 的协方差近似为：

$$
\widetilde C_{k,t}
=
Y_{k,t}B_{k,t}^{-1}Y_{k,t}^\top.
$$

算法只需要计算 $\widetilde C_{k,t}U$，其实现为：

$$
\widetilde C_{k,t}U
=
Y_{k,t}
B_{k,t}^{-1}
(Y_{k,t}^\top U).
$$

因此不构造 $n\times n$ 矩阵。

## 3.6 唯一旗目标

STRATA 在每次旗更新时最大化：

$$
\Phi_t(U)
=
\sum_{k=1}^{K}
\omega_k
\operatorname{tr}
\left[P_k(U)\widetilde C_{k,t}\right].
$$

令：

$$
A_{j,t}
=
\sum_{k=j}^{K}\omega_k\widetilde C_{k,t}.
$$

则：

$$
\Phi_t(U)
=
\sum_{j=1}^{K}
\operatorname{tr}
\left[U^{[j]\top}A_{j,t}U^{[j]}\right].
$$

其环境梯度按块为：

$$
H_t
=
2[
A_{1,t}U_t^{[1]},
\ldots,
A_{K,t}U_t^{[K]}
].
$$

这个目标是算法中唯一用于学习旗的目标。不存在额外的独立 PCA、周期 SVD 或层级重排步骤。

## 3.7 水平 Riemannian 梯度

旗流形采用商表示：

$$
\operatorname{Flag}(r_1,\ldots,r_K;n)
\simeq
\operatorname{St}(n,R)
/
[O(q_1)\times\cdots\times O(q_K)].
$$

先将环境梯度投影到 Stiefel 切空间：

$$
H_t^{\mathrm{St}}
=
H_t-U_t\operatorname{sym}(U_t^\top H_t).
$$

再去除块内旋转对应的垂直分量：

$$
H_t^{\mathrm{hor}}
=
H_t^{\mathrm{St}}
-
U_t\operatorname{blkdiag}
(U_t^\top H_t^{\mathrm{St}}).
$$

旗更新方向固定为梯度上升：

$$
\Delta_t=\xi H_t^{\mathrm{hor}}.
$$

## 3.8 QR retraction、块对齐与状态处理

旗只在：

$$
t\equiv0\pmod\tau
$$

时更新。先做 QR retraction：

$$
Q_tR_t=\operatorname{qr}(U_t+\Delta_t),
\qquad
\bar U_{t+1}=Q_t[:,1:R].
$$

QR 的符号通过强制 $R_t$ 对角线为正来固定。

然后对每个块做 Procrustes 对齐：

$$
L_{k,t}\Sigma_{k,t}R_{k,t}^\top
=
\bar U_{t+1}^{[k]\top}U_t^{[k]},
$$

$$
O_{k,t}=L_{k,t}R_{k,t}^\top,
$$

$$
U_{t+1}^{[k]}
=
\bar U_{t+1}^{[k]}O_{k,t}.
$$

低维一阶矩按相同坐标变换：

$$
M_{k,t}
\leftarrow
M_{k,t}O_{k,t}^\top.
$$

逐坐标二阶矩不是旋转等变的，因此算法不使用近似旋转或阈值规则。每次旗更新后固定执行：

$$
V_{k,t}\leftarrow0,
\qquad
c_{k,t}\leftarrow0.
$$

这一定义消除了二阶矩 transport 的歧义。下一步开始重新估计该块的二阶统计。

非旗更新步使用：

$$
U_{t+1}=U_t,
$$

且所有低维状态按普通 Adam 递推。

## 3.9 左侧旗

当 $m<n$ 时，对 $G_t^\top$ 应用上述右侧算法。等价地：

$$
U_t\in\operatorname{St}(m,R),
\qquad
Z_{k,t}=U_t^{[k]\top}G_t
\in\mathbb R^{q_k\times n},
$$

$$
D_{k,t}=U_t^{[k]}H_{k,t},
\qquad
C_t=G_tG_t^\top.
$$

其余状态更新、Nyström 统计、旗目标、retraction、对齐和二阶矩重置完全相同。

## 3.10 完整算法

```text
Algorithm 1: STRATA-AdamW for one matrix parameter W

Input:
    ranks 0 = r_0 < r_1 < ... < r_K = R
    timescales rho_1 > ... > rho_K
    block Adam constants beta1_k, beta2_k
    block scales a_k and residual scale a_perp
    learning-rate schedule eta_t, weight decay lambda_wd
    flag learning rate xi, flag interval tau
    fixed orthonormal sketches Omega_k

State:
    flag basis U = [U^[1], ..., U^[K]]
    first moments M_k = 0
    second moments V_k = 0
    counters c_k = 0
    covariance sketches Y_k = 0

for t = 0, ..., T - 1:
    G <- gradient of W

    # 1. Update multi-timescale covariance sketches.
    for k = 1, ..., K:
        Y_k <- rho_k Y_k + (1-rho_k) G^T (G Omega_k)

    # 2. Compute low-dimensional block gradients.
    for k = 1, ..., K:
        Z_k <- G U^[k]
        M_k <- beta1_k M_k + (1-beta1_k) Z_k
        V_k <- beta2_k V_k + (1-beta2_k) Z_k^2
        c_k <- c_k + 1
        Mhat_k <- M_k / (1-beta1_k^c_k)
        Vhat_k <- V_k / (1-beta2_k^c_k)
        H_k <- Mhat_k / (sqrt(Vhat_k) + epsilon)
        D_k <- H_k U^[k]^T

    # 3. Compute the stateless residual direction.
    G_perp <- G - sum_k (G U^[k]) U^[k]^T
    D_perp <- G_perp / (||G_perp||_F / sqrt(mn) + epsilon)

    # 4. Update the parameter.
    W <- (1-eta_t lambda_wd) W
         - eta_t [sum_k a_k D_k + a_perp D_perp]

    # 5. Update the flag every tau steps.
    if t mod tau == 0:
        for k = 1, ..., K:
            B_k <- sym(Omega_k^T Y_k) + gamma I
            define Ctilde_k X = Y_k B_k^{-1} (Y_k^T X)

        for j = 1, ..., K:
            H^[j] <- 2 sum_{k=j}^K omega_k Ctilde_k U^[j]

        H_st  <- H - U sym(U^T H)
        H_hor <- H_st - U blkdiag(U^T H_st)
        U_bar <- qf(U + xi H_hor)

        for k = 1, ..., K:
            O_k <- Procrustes(U_bar^[k], U^[k])
            U^[k] <- U_bar^[k] O_k
            M_k <- M_k O_k^T
            V_k <- 0
            c_k <- 0
```

## 3.11 单步算子形式

将低维 Adam 映射记为：

$$
\mathcal A_{k,t}(G_t;U_t,M_{k,t-1},V_{k,t-1})
=
H_{k,t}U_t^{[k]\top}.
$$

定义旗更新算子：

$$
\mathcal R_{\mathrm{Flag}}
(U_t,\{Y_{k,t}\})
=
\operatorname{Align}
\left[
\operatorname{qf}
\left(
U_t+\xi\operatorname{grad}\Phi_t(U_t)
\right),
U_t
\right].
$$

则 STRATA 的完整一步为：

$$
\boxed{
\begin{aligned}
Y_{k,t}
&=
\rho_kY_{k,t-1}
+(1-\rho_k)G_t^\top(G_t\Omega_k),
\\
D_t
&=
\sum_{k=1}^{K}
 a_k\mathcal A_{k,t}(G_t;U_t,M_{k,t-1},V_{k,t-1})
+
a_\perp
\frac{G_t(I-U_tU_t^\top)}
{\|G_t(I-U_tU_t^\top)\|_F/\sqrt{mn}+\epsilon},
\\
W_{t+1}
&=
(1-\eta_t\lambda_{\mathrm{wd}})W_t
-\eta_tD_t,
\\
U_{t+1}
&=
\begin{cases}
\mathcal R_{\mathrm{Flag}}(U_t,\{Y_{k,t}\}),
& t\equiv0\pmod\tau,\\
U_t,
& \text{otherwise}.
\end{cases}
\end{aligned}
}
$$

这四行定义了主算法的全部计算路径。

## 3.12 状态与计算复杂度

右侧旗的低维 Adam 状态大小为：

$$
2\sum_{k=1}^{K}mq_k=2mR.
$$

旗基大小为：

$$
nR.
$$

Nyström sketch 大小为：

$$
\sum_{k=1}^{K}ns_k.
$$

因此每个矩阵参数的总附加状态为：

$$
\boxed{
\operatorname{Mem}_{\mathrm{STRATA}}
=
2mR+nR+n\sum_{k=1}^{K}(r_k+p)
}
$$

个标量，而 full AdamW 需要约：

$$
\operatorname{Mem}_{\mathrm{AdamW}}=2mn
$$

个一阶与二阶状态标量。

每个普通训练步的主要附加计算为：

$$
O(mnR)
+
O\left(mn\sum_{k=1}^{K}s_k\right)
$$

中的矩阵乘法；其中 sketch 更新可以与梯度投影批量合并。每次旗更新还需要：

$$
O\left(
\sum_{k=1}^{K}
(ns_k^2+s_k^3+ns_kR)
+nR^2
\right).
$$

由于旗每 $\tau$ 步更新一次，其摊销开销除以 $\tau$。

---

# 4. 深入理论

理论部分必须包含五条完整主线：

1. **旗空间最优性**；
2. **流形优化 landscape**；
3. **参数优化 convergence**；
4. **子空间估计与更新 error bound**；
5. **generalization bound**。

所有结果需要明确区分：

- exact covariance；
- stochastic covariance；
- sketched covariance；
- stationary setting；
- drifting setting；
- exact flag minimizer；
- online approximate flag iterate。

---

## 4.1 空间最优性

### 4.1.1 联合旗目标

定义：

$$
\Phi(U)
=
\sum_{k=1}^{K}
\omega_k
\operatorname{tr}
\left[
P_k(U)C_k
\right].
$$

最大化 $\Phi$ 等价于最小化多时间尺度重构误差。

令：

$$
A_j
=
\sum_{k=j}^{K}
\omega_kC_k.
$$

由于第 $j$ 个块进入所有 $P_k$，$k\ge j$，有：

$$
\Phi(U)
=
\sum_{j=1}^{K}
\operatorname{tr}
\left[
U^{[j]\top}A_jU^{[j]}
\right].
$$

这将旗目标写为带块正交约束的加权 trace maximization。

### Theorem 1：共享特征基下的全局最优旗

假设所有 $C_k$ 两两可交换：

$$
C_iC_j=C_jC_i.
$$

则存在共同正交特征基：

$$
C_k
=
V\Lambda_kV^\top.
$$

定义每个共同特征向量 $v_i$ 在块 $j$ 中的累计收益：

$$
s_{j,i}
=
\sum_{k=j}^{K}
\omega_k\lambda_{k,i}.
$$

若存在一个分块索引划分：

$$
I_1,\ldots,I_K,
\qquad
|I_j|=q_j,
$$

使任何跨块交换都不增加：

$$
\sum_{j=1}^{K}\sum_{i\in I_j}s_{j,i},
$$

则：

$$
U^{[j]\star}
=
V_{I_j}
$$

构成全局最优旗。

若收益严格分离：

$$
\min_{i\in I_j}s_{j,i}
>
\max_{i\in I_\ell}s_{j,i},
\qquad
\ell>j,
$$

则最优旗在块内旋转意义下唯一。

该定理给出一个可计算的空间最优解，并明确什么时候长期核心方向应进入内层块。

### Theorem 2：非交换协方差下的近似最优性

设：

$$
\bar C_k
$$

是一组可交换近似，且：

$$
\|C_k-\bar C_k\|_2\le\varepsilon_k.
$$

令 $\bar U^\star$ 为可交换系统的最优旗，$U^\star$ 为真实最优旗。则目标差满足：

$$
0
\le
\Phi(U^\star)-\Phi(\bar U^\star)
\le
2\sum_{k=1}^{K}
\omega_kr_k\varepsilon_k.
$$

如果共享基近似误差小，则共同特征基下构造的旗接近真实最优。

### Theorem 3：与单一 Grassmann 的预算化比较

单一秩 $R=r_K$ 子空间使用：

$$
P\in\operatorname{Gr}(R,d).
$$

定义成本化目标：

$$
J_{\mathrm{single}}(P)
=
\sum_{k=1}^{K}
\omega_k
\operatorname{tr}[(I-P)C_k]
+
\sum_{k=1}^{K}
\lambda_kr_K.
$$

旗目标：

$$
J_{\mathrm{flag}}(U)
=
\sum_{k=1}^{K}
\omega_k
\operatorname{tr}[(I-P_k)C_k]
+
\sum_{k=1}^{K}
\lambda_kr_k.
$$

若第 $k$ 时间尺度的谱尾满足：

$$
\sum_{i>r_k}\lambda_i(C_k)
-
\sum_{i>R}\lambda_i(C_k)
<
\lambda_k(R-r_k),
$$

则使用较小内层子空间的误差增加，小于其状态成本下降，因此：

$$
\min_UJ_{\mathrm{flag}}(U)
<
\min_PJ_{\mathrm{single}}(P).
$$

这个结果不使用错误的“更小秩投影误差一定更低”结论，而是在相同优化器状态成本下比较误差与内存。

---

## 4.2 Landscape 理论

### 4.2.1 一阶临界点

Riemannian 一阶条件：

$$
\operatorname{grad}\Phi(U)=0.
$$

对块 $j$，等价于：

$$
(I-UU^\top)A_jU^{[j]}
+
\text{cross-block horizontal terms}
=
0.
$$

在共享特征基条件下，所有临界旗均由共同特征向量分块组成。

### Theorem 4：共享特征基下无伪局部极大值

假设 $C_k$ 两两可交换，且所有相关累计收益 $s_{j,i}$ 无相等值。则 $\Phi$ 在旗流形上的每个局部极大值都是全局极大值；任何非最优临界旗都有严格正曲率上升方向，等价地，对最小化 $-\Phi$ 来说存在严格负曲率方向。

证明使用跨块特征向量交换。若非最优旗把较高收益方向放在外层、较低收益方向放在内层，则沿两维旋转测地线可得到：

$$
\frac{d^2}{d\alpha^2}
\Phi(U(\alpha))
\bigg|_{\alpha=0}
>
0.
$$

因此非最优临界点为 strict saddle。

### Corollary 4.1：随机初始化逃离 saddle

在光滑性、噪声非退化和 strict-saddle 条件下，带随机扰动的 Riemannian gradient 方法以高概率避开非最优临界点，并收敛到全局最优旗的邻域。

### Theorem 5：近交换条件下的 landscape 稳定性

设：

$$
C_k=\bar C_k+E_k,
$$

其中 $\bar C_k$ 两两可交换，且：

$$
\|E_k\|_2\le\varepsilon.
$$

若可交换系统的最小累计 eigengap 为：

$$
\Delta_{\min}>0,
$$

且：

$$
\varepsilon
<
c\Delta_{\min},
$$

则：

1. 全局最优旗偏移为：

$$
d_{\mathrm{Flag}}(U^\star,\bar U^\star)
=
O\left(\frac{\varepsilon}{\Delta_{\min}}\right);
$$

2. 原 strict saddle 的负曲率只发生 $O(\varepsilon)$ 扰动；
3. 不会在远离真实最优旗的区域产生高质量伪局部极大值。

该结果可由 Davis–Kahan 型子空间扰动和 Riemannian Hessian perturbation 组合得到。

---

## 4.3 参数优化 convergence

### 4.3.1 分层预条件形式

STRATA 参数更新可写为：

$$
\theta_{t+1}
=
\theta_t-B_tg_t,
$$

其中：

$$
B_t
=
\sum_{k=1}^{K}
\eta_{k,t}\Pi_{k,t}
+
\eta_{\perp,t}\Pi_{\perp,t}.
$$

若：

$$
0<\eta_{\min,t}
\le
\eta_{k,t},\eta_{\perp,t}
\le
\eta_{\max,t},
$$

则：

$$
\eta_{\min,t}I
\preceq
B_t
\preceq
\eta_{\max,t}I.
$$

### Assumptions

**A1. Smoothness**

$$
\|\nabla f(\theta)-\nabla f(\theta')\|
\le
L\|\theta-\theta'\|.
$$

**A2. Lower bounded objective**

$$
f(\theta)\ge f_\star.
$$

**A3. Unbiased stochastic gradients**

$$
\mathbb E[g_t\mid\mathcal F_t]
=
\nabla f(\theta_t).
$$

**A4. Bounded conditional variance**

$$
\mathbb E[
\|g_t-\nabla f(\theta_t)\|^2
\mid\mathcal F_t
]
\le\sigma^2.
$$

**A5. Delayed flag state**

$B_t$ 由 $t-1$ 及以前的信息构造，避免当前梯度和投影估计之间的依赖偏差。

### Theorem 6：非凸 stationarity

若将 $B_t=\eta_tA_t$ 写成基础步长与层级算子的乘积，并满足：

$$
a_{\min}I\preceq A_t\preceq a_{\max}I,
\qquad
L\eta_ta_{\max}^2\le a_{\min},
$$

则：

$$
\sum_{t=0}^{T-1}
\eta_{\min,t}
\mathbb E
\|\nabla f(\theta_t)\|^2
\le
2(f(\theta_0)-f_\star)
+
L\sigma^2
\sum_{t=0}^{T-1}
\eta_{\max,t}^2.
$$

若：

$$
\eta_{k,t}=a_k\eta,
\qquad
\eta_{\perp,t}=a_\perp\eta,
$$

且：

$$
\eta=\Theta(T^{-1/2}),
$$

则随机迭代点 $R$ 满足：

$$
\mathbb E
\|\nabla f(\theta_R)\|^2
=
O(T^{-1/2}).
$$

### Theorem 7：带旗估计误差的 descent bound

令理想预条件器为：

$$
B_t^\star
=
\sum_k\eta_{k,t}\Pi_{k,t}^\star
+
\eta_{\perp,t}\Pi_{\perp,t}^\star.
$$

实际误差：

$$
E_t=B_t-B_t^\star.
$$

则一步下降额外项可由：

$$
|\langle\nabla f,E_t\nabla f\rangle|
\le
\|E_t\|_2
\|\nabla f\|^2
$$

控制。

如果：

$$
\|E_t\|_2
\le
\sum_{k=1}^{K}
|\eta_{k,t}-\eta_{k+1,t}|
\|
P_{k,t}-P_{k,t}^\star
\|_2,
$$

则 convergence 常数直接由各层子空间估计误差和相邻层学习率差控制。

该结果说明：若不同层学习率相同，则旗误差对参数更新影响减弱；若学习率差较大，则需要更准确的旗估计。

### Theorem 8：PL 条件下线性收敛到噪声邻域

若目标满足 Polyak–Łojasiewicz 条件：

$$
\frac12
\|\nabla f(\theta)\|^2
\ge
\mu(f(\theta)-f_\star),
$$

并且：

$$
mI\preceq B_t\preceq MI,
$$

则：

$$
\mathbb E[f(\theta_{t+1})-f_\star]
\le
(1-2\mu m+L\mu M^2)
\mathbb E[f(\theta_t)-f_\star]
+
\frac{L}{2}M^2\sigma^2.
$$

在确定性梯度下得到线性收敛；在随机梯度下收敛到由 $\sigma^2$ 决定的邻域。

---

## 4.4 子空间 error bounds

### 4.4.1 有限样本协方差误差

对时间尺度 $k$，设真实协方差为：

$$
\Sigma_{k,t}
=
\mathbb E[C_{k,t}].
$$

估计误差：

$$
\widehat C_{k,t}-\Sigma_{k,t}.
$$

在次高斯梯度和有效样本数：

$$
N_{\mathrm{eff},k}
\approx
\frac{1+\rho_k}{1-\rho_k}
$$

下，矩阵 Bernstein 给出高概率界：

$$
\|
\widehat C_{k,t}-\Sigma_{k,t}
\|_2
\le
C
\left[
\sqrt{
\frac{
\|\Sigma_{k,t}\|_2
\log(d/\delta)
}{
N_{\mathrm{eff},k}
}
}
+
\frac{
G_{\max}^2\log(d/\delta)
}{
N_{\mathrm{eff},k}
}
\right].
$$

### Theorem 9：单层 Davis–Kahan error bound

若第 $k$ 层真实协方差 eigengap：

$$
\Delta_{k,t}
=
\lambda_{r_k}(\Sigma_{k,t})
-
\lambda_{r_k+1}(\Sigma_{k,t})
>0,
$$

则：

$$
\|
\sin\Theta(
\widehat{\mathcal V}_{k,t},
\mathcal V_{k,t}^\star
)
\|_2
\le
\frac{
\|\widehat C_{k,t}-\Sigma_{k,t}\|_2
}{
\Delta_{k,t}
}.
$$

### Theorem 10：联合旗误差

定义旗距离：

$$
d_{\mathrm{Flag}}^2(
\widehat U_t,U_t^\star
)
=
\sum_{k=1}^{K}
\alpha_k
\|
\widehat P_{k,t}-P_{k,t}^\star
\|_F^2.
$$

若联合目标在最优旗附近具有 geodesic strong concavity 常数 $\mu_{\mathrm{flag}}$，且估计目标梯度误差为：

$$
\|
\operatorname{grad}\widehat\Phi_t
-
\operatorname{grad}\Phi_t
\|
\le\varepsilon_t,
$$

则局部最优估计满足：

$$
d_{\mathrm{Flag}}(
\widehat U_t,U_t^\star
)
\le
\frac{2\varepsilon_t}{\mu_{\mathrm{flag}}}.
$$

将协方差误差代入可得到：

$$
d_{\mathrm{Flag}}
=
O\left(
\sum_k
\frac{
\omega_k
\|\widehat C_{k,t}-\Sigma_{k,t}\|_2
}{
\mu_{\mathrm{flag}}
}
\right).
$$

### 4.4.2 Sketch error

若使用随机 range sketch，且 oversampling 为 $p$，则每层协方差近似误差满足标准随机低秩近似形式：

$$
\|C_k-\widetilde C_k\|_2
\le
(1+\epsilon_{\mathrm{sk}})
\lambda_{r_k+1}(C_k)
$$

以高概率成立。

联合旗目标误差：

$$
|\Phi_C(U)-\Phi_{\widetilde C}(U)|
\le
\sum_{k=1}^{K}
\omega_kr_k
\|C_k-\widetilde C_k\|_2.
$$

因此 sketch 引起的最优值误差为：

$$
\Phi_C(U_C^\star)
-
\Phi_C(U_{\widetilde C}^\star)
\le
2
\sum_{k=1}^{K}
\omega_kr_k
\|C_k-\widetilde C_k\|_2.
$$

### Theorem 11：漂移环境中的 tracking bound

设真实最优旗漂移：

$$
d_{\mathrm{Flag}}(
U_{t+1}^\star,U_t^\star
)
\le\delta_t.
$$

在线更新：

$$
U_{t+1}
=
\operatorname{Retr}_{U_t}
\left(
\xi_t\widehat{\operatorname{grad}}\Phi_t(U_t)
\right).
$$

在局部 strong concavity、gradient Lipschitz 和有界随机误差条件下：

$$
\mathbb E[
d_{\mathrm{Flag}}(U_{t+1},U_{t+1}^\star)
]
\le
(1-\mu_{\mathrm{flag}}\xi_t)
\mathbb E[
d_{\mathrm{Flag}}(U_t,U_t^\star)
]
+
\xi_t\varepsilon_t
+
\delta_t
+
O(\xi_t^2).
$$

固定小步长下稳态误差：

$$
\limsup_t
\mathbb E[d_{\mathrm{Flag}}]
=
O\left(
\frac{
\xi\varepsilon+\delta
}{
\mu_{\mathrm{flag}}\xi
}
\right).
$$

该式明确给出噪声—响应速度折中。

---

## 4.5 Generalization 理论

STRATA 不改变最终假设类，因此不能直接声称“使用旗流形就缩小模型函数类”。generalization 分析必须通过**算法稳定性、更新能量和数据依赖有效维数**建立。

### 4.5.1 邻接数据集稳定性

令训练集 $S$ 和 $S'$ 只差一个样本，对应迭代：

$$
\theta_t,
\qquad
\theta_t'.
$$

更新：

$$
\theta_{t+1}
=
\theta_t-B_tg_t,
$$

$$
\theta_{t+1}'
=
\theta_t'-B_t'g_t'.
$$

定义：

$$
\Delta_t
=
\mathbb E\|\theta_t-\theta_t'\|.
$$

若单样本梯度为 $L_g$-Lipschitz，且：

$$
\|B_t\|_2\le M_t,
$$

$$
\|B_t-B_t'\|_2
\le
L_B\|\theta_t-\theta_t'\|
+
\zeta_t,
$$

其中 $\zeta_t$ 表示旗估计对单样本替换的敏感度，则：

$$
\Delta_{t+1}
\le
\left[
1+M_tL_g+L_BG
\right]\Delta_t
+
\frac{2M_tG}{N}
+
G\zeta_t.
$$

### Theorem 12：uniform stability bound

若：

$$
M_tL_g+L_BG\le c_t,
$$

则：

$$
\epsilon_{\mathrm{stab}}
\le
L_\ell
\sum_{t=0}^{T-1}
\left(
\frac{2M_tG}{N}
+
G\zeta_t
\right)
\prod_{j=t+1}^{T-1}(1+c_j).
$$

旗状态的样本敏感度可进一步由协方差子空间扰动控制：

$$
\zeta_t
\le
\sum_{k=1}^{K}
|\eta_{k,t}-\eta_{k+1,t}|
\frac{
C
}{
N_{\mathrm{eff},k}\Delta_{k,t}
}.
$$

因此，较大 eigengap、较长时间平均和较平滑的层级学习率差都会改善稳定性界。

### 4.5.2 数据依赖有效维数

定义时间尺度 $k$ 的有效维数：

$$
d_{\mathrm{eff},k}(\lambda)
=
\operatorname{tr}
\left[
C_k(C_k+\lambda I)^{-1}
\right].
$$

若梯度能量主要集中在小的嵌套子空间，则：

$$
d_{\mathrm{eff},1}
\le
d_{\mathrm{eff},2}
\le
\cdots
\le
d.
$$

对层级投影后的随机更新噪声：

$$
\mathbb E
\|B_t(g_t-\nabla f_t)\|^2
=
\sum_{k=1}^{K}
\eta_{k,t}^2
\operatorname{tr}
\left(
\Pi_{k,t}\Sigma_t
\right)
+
\eta_{\perp,t}^2
\operatorname{tr}
\left(
\Pi_{\perp,t}\Sigma_t
\right).
$$

### Theorem 13：更新噪声 generalization proxy

在 smooth convex 或局部稳定区域，on-average stability 可由累计更新噪声控制：

$$
\epsilon_{\mathrm{gen}}
\le
\frac{C}{N}
\sum_{t=0}^{T-1}
\left[
\sum_{k=1}^{K}
\eta_{k,t}^2
\operatorname{tr}(\Pi_{k,t}\Sigma_t)
+
\eta_{\perp,t}^2
\operatorname{tr}(\Pi_{\perp,t}\Sigma_t)
\right]^{1/2}.
$$

若 STRATA 对高方差快速层使用较短记忆或受限步长，而稳定层使用较长平均，则该项可能小于统一低秩或 full-rank 更新。

该结果不会被表述为无条件更好的 generalization；实验必须测量该界中的各项。

---


## 4.6 统一误差分解与端到端理论

前述结果分别讨论空间最优性、landscape、参数收敛、子空间误差和 generalization。为了形成完整理论，需要把这些模块连接成同一个耦合系统。STRATA 的实际更新不是使用真实协方差上的精确最优旗，而是依次经过：

$$
\Sigma_{k,t}
\longrightarrow
\widehat C_{k,t}
\longrightarrow
\widetilde C_{k,t}
\longrightarrow
\widetilde{\mathcal U}_{t}^{\star}
\longrightarrow
\mathcal U_t
\longrightarrow
B_t
\longrightarrow
\theta_{t+1}.
$$

其中：

- $\Sigma_{k,t}$ 是第 $k$ 个时间尺度的 population covariance；
- $\widehat C_{k,t}$ 是指数滑动经验协方差；
- $\widetilde C_{k,t}$ 是 sketch 后协方差；
- $\widetilde{\mathcal U}_{t}^{\star}$ 是 sketch 目标的精确最优旗；
- $\mathcal U_t$ 是在线 Riemannian 迭代得到的实际旗；
- $B_t$ 是实际层级预条件器。

因此总误差必须分为：

$$
\boxed{
\mathcal E_t^{\mathrm{total}}
=
\mathcal E_t^{\mathrm{stat}}
+
\mathcal E_t^{\mathrm{sketch}}
+
\mathcal E_t^{\mathrm{flag-opt}}
+
\mathcal E_t^{\mathrm{drift}}
+
\mathcal E_t^{\mathrm{transport}}
}
$$

### 4.6.1 Population、empirical 与 sketched 目标

定义：

$$
\Phi_t(U)
=
\sum_{k=1}^{K}
\omega_k\operatorname{tr}(P_k(U)\Sigma_{k,t}),
$$

$$
\widehat\Phi_t(U)
=
\sum_{k=1}^{K}
\omega_k\operatorname{tr}(P_k(U)\widehat C_{k,t}),
$$

$$
\widetilde\Phi_t(U)
=
\sum_{k=1}^{K}
\omega_k\operatorname{tr}(P_k(U)\widetilde C_{k,t}).
$$

令：

$$
U_t^{\star}
\in
\arg\max_U\Phi_t(U),
$$

$$
\widehat U_t^{\star}
\in
\arg\max_U\widehat\Phi_t(U),
$$

$$
\widetilde U_t^{\star}
\in
\arg\max_U\widetilde\Phi_t(U).
$$

定义每层统计误差和 sketch 误差：

$$
e_{k,t}^{\mathrm{stat}}
=
\|\widehat C_{k,t}-\Sigma_{k,t}\|_2,
$$

$$
e_{k,t}^{\mathrm{sk}}
=
\|\widetilde C_{k,t}-\widehat C_{k,t}\|_2.
$$

则对任意可行旗 $U$：

$$
|\widehat\Phi_t(U)-\Phi_t(U)|
\le
\sum_{k=1}^{K}\omega_kr_ke_{k,t}^{\mathrm{stat}},
$$

$$
|\widetilde\Phi_t(U)-\widehat\Phi_t(U)|
\le
\sum_{k=1}^{K}\omega_kr_ke_{k,t}^{\mathrm{sk}}.
$$

因此最优值误差满足：

$$
\boxed{
0
\le
\Phi_t(U_t^{\star})-
\Phi_t(\widetilde U_t^{\star})
\le
2\sum_{k=1}^{K}
\omega_kr_k
\left(
e_{k,t}^{\mathrm{stat}}+e_{k,t}^{\mathrm{sk}}
\right)
}
$$

### Theorem 14：联合旗扰动界

假设 $\Phi_t$ 在 $U_t^{\star}$ 的测地邻域内具有 geodesic strong concavity 常数 $\mu_{\mathrm{flag},t}>0$，并且相关累计 eigengap 为 $\Delta_{\mathrm{flag},t}>0$。则存在常数 $C>0$，使：

$$
\boxed{
 d_{\mathrm{Flag}}
(\widetilde U_t^{\star},U_t^{\star})
\le
\frac{C}{\mu_{\mathrm{flag},t}}
\sum_{k=1}^{K}
\omega_k
\left(
e_{k,t}^{\mathrm{stat}}+e_{k,t}^{\mathrm{sk}}
\right)
}
$$

在共享特征基或近交换条件下，$\mu_{\mathrm{flag},t}$ 与最小累计 eigengap 同阶，因此：

$$
 d_{\mathrm{Flag}}
(\widetilde U_t^{\star},U_t^{\star})
=
O\left(
\frac{
\sum_k\omega_k(e_{k,t}^{\mathrm{stat}}+e_{k,t}^{\mathrm{sk}})
}{
\Delta_{\mathrm{flag},t}
}
\right).
$$

### 4.6.2 Online flag tracking

实际旗更新为：

$$
U_{t+1}
=
\operatorname{Retr}_{U_t}
\left(
\xi_t\widehat h_t
\right),
$$

其中：

$$
\mathbb E[
\widehat h_t\mid\mathcal F_t
]
=
\operatorname{grad}\widetilde\Phi_t(U_t)+b_t,
$$

$$
\mathbb E[
\|\widehat h_t-
\mathbb E[\widehat h_t\mid\mathcal F_t]\|^2
\mid\mathcal F_t
]
\le
\sigma_{\mathrm{flag},t}^2.
$$

定义最优旗漂移：

$$
\delta_t
=
 d_{\mathrm{Flag}}(U_{t+1}^{\star},U_t^{\star}).
$$

定义实际 tracking error：

$$
e_t^{\mathrm{track}}
=
\mathbb E[
 d_{\mathrm{Flag}}(U_t,U_t^{\star})
].
$$

### Theorem 15：一步 tracking 递推

若 sketch 目标在当前邻域内具有 geodesic strong concavity 常数 $\mu_t$，Riemannian gradient 为 $L_{\mathrm{flag},t}$-Lipschitz，且：

$$
0<\xi_t\le\frac{c}{L_{\mathrm{flag},t}},
$$

则：

$$
\boxed{
\begin{aligned}
e_{t+1}^{\mathrm{track}}
\le
&(1-\mu_t\xi_t)e_t^{\mathrm{track}}\\
&+
C\xi_t
\sum_{k=1}^{K}
\omega_k
\left(
e_{k,t}^{\mathrm{stat}}+e_{k,t}^{\mathrm{sk}}
\right)\\
&+
C\xi_t\|b_t\|
+
C\xi_t^2\sigma_{\mathrm{flag},t}^2
+
\delta_t.
\end{aligned}
}
$$

递归展开：

$$
\boxed{
\begin{aligned}
e_T^{\mathrm{track}}
\le
&
\left[
\prod_{s=0}^{T-1}(1-\mu_s\xi_s)
\right]
e_0^{\mathrm{track}}\\
&+
\sum_{t=0}^{T-1}
\Bigg[
C\xi_t
\sum_{k=1}^{K}
\omega_k
(e_{k,t}^{\mathrm{stat}}+e_{k,t}^{\mathrm{sk}})
+
C\xi_t\|b_t\|\\
&\qquad
g+
C\xi_t^2\sigma_{\mathrm{flag},t}^2
+
\delta_t
\Bigg]
\prod_{s=t+1}^{T-1}(1-\mu_s\xi_s).
\end{aligned}
}
$$

固定 $\mu,
\xi$ 且各误差一致有界时：

$$
\boxed{
\limsup_{t\to\infty}e_t^{\mathrm{track}}
=
O\left(
\frac{e^{\mathrm{stat}}+e^{\mathrm{sk}}+\|b\|}{\mu}
+
\frac{\xi\sigma_{\mathrm{flag}}^2}{\mu}
+
\frac{\delta}{\mu\xi}
\right)
}
$$

该式给出 flag learning rate 的噪声—漂移折中。平衡后：

$$
\xi_{\mathrm{opt}}
\asymp
\sqrt{
\frac{\delta}{\sigma_{\mathrm{flag}}^2}
},
$$

$$
\limsup_t e_t^{\mathrm{track}}
=
O\left(
\frac{e^{\mathrm{stat}}+e^{\mathrm{sk}}+\|b\|}{\mu}
+
\frac{\sqrt{\delta\sigma_{\mathrm{flag}}^2}}{\mu}
\right).
$$

### 4.6.3 Flag error 到预条件器 error

STRATA 线性化预条件器为：

$$
B_t
=
\eta_t
\left[
\sum_{k=1}^{K}a_{k,t}\Pi_{k,t}
+
a_{\perp,t}\Pi_{\perp,t}
\right].
$$

令：

$$
a_{K+1,t}=a_{\perp,t}.
$$

利用：

$$
\Pi_{k,t}=P_{k,t}-P_{k-1,t},
$$

可写成 telescoping 形式：

$$
\boxed{
B_t
=
\eta_t
\left[
a_{\perp,t}I
+
\sum_{k=1}^{K}
(a_{k,t}-a_{k+1,t})P_{k,t}
\right]
}
$$

设理想预条件器为：

$$
B_t^{\star}
=
\eta_t
\left[
a_{\perp,t}I
+
\sum_{k=1}^{K}
(a_{k,t}-a_{k+1,t})P_{k,t}^{\star}
\right].
$$

### Theorem 16：预条件器 perturbation

定义层级差异强度：

$$
\Gamma_t^a
=
\sum_{k=1}^{K}
|a_{k,t}-a_{k+1,t}|.
$$

则：

$$
\boxed{
\|B_t-B_t^{\star}\|_2
\le
\eta_t
\sum_{k=1}^{K}
|a_{k,t}-a_{k+1,t}|
\|P_{k,t}-P_{k,t}^{\star}\|_2
\le
C\eta_t\Gamma_t^a e_t^{\mathrm{track}}
}
$$

该结果有两个直接结论。

第一，旗误差只有在相邻层更新系数不同的时候才影响参数方向。第二，若：

$$
a_{1,t}=\cdots=a_{K,t}=a_{\perp,t},
$$

则：

$$
\Gamma_t^a=0,
$$

旗几何从线性更新中完全消失。这严格对应“所有层级学习率相同”的消融。

### 4.6.4 修正后的非凸 convergence

原 Theorem 6 中的步长条件：

$$
\eta_{\max,t}
\le
\frac{\eta_{\min,t}}{L\eta_{\max,t}}
$$

记号不一致，也把已经包含步长的量再次比较。统一写成：

$$
B_t
=
\eta_tA_t,
$$

并假设：

$$
a_{\min}I
\preceq
A_t
\preceq
a_{\max}I.
$$

正确的小步长条件为：

$$
\boxed{
L\eta_ta_{\max}^2
\le
a_{\min}
}
$$

参数更新进一步写为：

$$
\theta_{t+1}
=
\theta_t-B_tg_t+r_t,
$$

其中 $r_t$ 汇总以下误差：

$$
r_t
=
r_t^{\mathrm{transport}}
+
r_t^{\mathrm{second-moment}}
+
r_t^{\mathrm{clipping}}
+
r_t^{\mathrm{numerical}}.
$$

### Theorem 17：带 flag 与 transport error 的非凸 stationarity

假设 $f$ 为 $L$-smooth、下有界，随机梯度无偏且：

$$
\mathbb E[
\|g_t-\nabla f(\theta_t)\|^2
\mid\mathcal F_t
]
\le
\sigma_t^2.
$$

若 $A_t$ 在抽取当前 minibatch 前确定，并满足上述谱界和步长条件，则令：

$$
S_T
=
\sum_{t=0}^{T-1}\eta_t,
$$

有：

$$
\boxed{
\begin{aligned}
\frac{1}{S_T}
\sum_{t=0}^{T-1}
\eta_t
\mathbb E\|\nabla f(\theta_t)\|^2
\le
&
\frac{2[f(\theta_0)-f_\star]}
{a_{\min}S_T}\\
&+
\frac{La_{\max}^2}{a_{\min}}
\frac{
\sum_{t=0}^{T-1}
\eta_t^2\sigma_t^2
}{S_T}\\
&+
\frac{C}{S_T}
\sum_{t=0}^{T-1}
\eta_t\Gamma_t^a e_t^{\mathrm{track}}\\
&+
\frac{C}{S_T}
\sum_{t=0}^{T-1}
\left[
\mathbb E\|\nabla f(\theta_t)\|\|r_t\|
+
L\mathbb E\|r_t\|^2
\right].
\end{aligned}
}
$$

其中第三项是旗估计引起的额外预条件器误差，第四项是 transport 与低维自适应状态近似误差。

若：

$$
\eta_t=\Theta(T^{-1/2}),
$$

$$
\sup_t\sigma_t^2<\infty,
$$

$$
\frac{1}{S_T}
\sum_t\eta_t\Gamma_t^ae_t^{\mathrm{track}}
=o(1),
$$

$$
\frac{1}{S_T}
\sum_t
\left[
\mathbb E\|\nabla f(\theta_t)\|\|r_t\|
+
L\mathbb E\|r_t\|^2
\right]
=o(1),
$$

则随机迭代点 $R$ 满足：

$$
\mathbb E\|\nabla f(\theta_R)\|^2
=
O(T^{-1/2})+o(1).
$$

### 4.6.5 PL 条件下的完整递推

若：

$$
\frac12\|\nabla f(\theta)\|^2
\ge
\mu[f(\theta)-f_\star],
$$

则：

$$
\boxed{
\begin{aligned}
\mathbb E[f(\theta_{t+1})-f_\star]
\le
&(1-\mu\eta_ta_{\min})
\mathbb E[f(\theta_t)-f_\star]\\
&+
\frac{L\eta_t^2a_{\max}^2}{2}
\sigma_t^2\\
&+
C\eta_t\Gamma_t^a e_t^{\mathrm{track}}\\
&+
C\mathbb E
\left[
\|\nabla f(\theta_t)\|\|r_t\|
+L\|r_t\|^2
\right].
\end{aligned}
}
$$

常数步长下展开为：

$$
\begin{aligned}
\mathbb E[f(\theta_T)-f_\star]
\le
&
(1-\mu\eta a_{\min})^T
[f(\theta_0)-f_\star]\\
&+
\sum_{t=0}^{T-1}
(1-\mu\eta a_{\min})^{T-1-t}
\Bigg[
\frac{L\eta^2a_{\max}^2}{2}\sigma_t^2\\
&
\qquad+
C\eta\Gamma_t^ae_t^{\mathrm{track}}
+
C\mathbb E(
\|\nabla f_t\|\|r_t\|+L\|r_t\|^2
)
\Bigg].
\end{aligned}
$$

因此确定性、静态、exact-sketch、exact-transport 情况下恢复线性收敛；随机、漂移和近似实现情况下收敛到由各误差共同决定的邻域。

---

## 4.7 最终总定理与完整大公式

### Theorem 18：STRATA 端到端 stationarity bound

在 Theorems 14–17 的条件下，定义：

$$
q_t
=
1-\mu_t\xi_t,
$$

$$
\mathcal C_t
=
\sum_{k=1}^{K}
\omega_k
\left(
e_{k,t}^{\mathrm{stat}}+e_{k,t}^{\mathrm{sk}}
\right),
$$

$$
\mathcal G_t
=
\xi_t\mathcal C_t
+
\xi_t\|b_t\|
+
\xi_t^2\sigma_{\mathrm{flag},t}^2
+
\delta_t.
$$

则：

$$
 e_t^{\mathrm{track}}
\le
\left(
\prod_{s=0}^{t-1}q_s
\right)e_0^{\mathrm{track}}
+
C
\sum_{j=0}^{t-1}
\mathcal G_j
\prod_{s=j+1}^{t-1}q_s.
$$

将该式代入 Theorem 17，得到完整端到端界：

$$
\boxed{
\begin{aligned}
&
\frac{1}{S_T}
\sum_{t=0}^{T-1}
\eta_t
\mathbb E\|\nabla f(\theta_t)\|^2
\\[2mm]
&\le
\underbrace{
\frac{2[f(\theta_0)-f_\star]}
{a_{\min}S_T}
}_{\text{parameter initialization}}
+
\underbrace{
\frac{La_{\max}^2}{a_{\min}}
\frac{\sum_{t=0}^{T-1}\eta_t^2\sigma_t^2}{S_T}
}_{\text{stochastic gradient noise}}
\\[2mm]
&\quad+
\underbrace{
\frac{C}{S_T}
\sum_{t=0}^{T-1}
\eta_t\Gamma_t^a
\left(
\prod_{s=0}^{t-1}q_s
\right)e_0^{\mathrm{track}}
}_{\text{flag initialization}}
\\[2mm]
&\quad+
\underbrace{
\frac{C}{S_T}
\sum_{t=0}^{T-1}
\eta_t\Gamma_t^a
\sum_{j=0}^{t-1}
\left[
\xi_j
\sum_{k=1}^{K}
\omega_k
\left(
e_{k,j}^{\mathrm{stat}}+e_{k,j}^{\mathrm{sk}}
\right)
\right]
\prod_{s=j+1}^{t-1}q_s
}_{\text{covariance estimation and sketch approximation}}
\\[2mm]
&\quad+
\underbrace{
\frac{C}{S_T}
\sum_{t=0}^{T-1}
\eta_t\Gamma_t^a
\sum_{j=0}^{t-1}
\left[
\xi_j\|b_j\|
+
\xi_j^2\sigma_{\mathrm{flag},j}^2
\right]
\prod_{s=j+1}^{t-1}q_s
}_{\text{online Riemannian optimization}}
\\[2mm]
&\quad+
\underbrace{
\frac{C}{S_T}
\sum_{t=0}^{T-1}
\eta_t\Gamma_t^a
\sum_{j=0}^{t-1}
\delta_j
\prod_{s=j+1}^{t-1}q_s
}_{\text{nonstationary flag drift}}
\\[2mm]
&\quad+
\underbrace{
\frac{C}{S_T}
\sum_{t=0}^{T-1}
\left[
\mathbb E\|\nabla f(\theta_t)\|\|r_t\|
+
L\mathbb E\|r_t\|^2
\right]
}_{\text{transport, second-moment, clipping, and numerical error}}.
\end{aligned}
}
$$

该公式是 STRATA 理论部分的最终结论。它把最终参数 stationarity 分成七个可单独测量的来源：

$$
\boxed{
\begin{array}{c}
\text{parameter initialization}
+
\text{SGD noise}
+
\text{flag initialization}
+
\text{statistical/sketch error}
+
\text{flag optimization error}
+
\text{drift error}
+
\text{state transport error}
\end{array}
}
$$

### Corollary 18.1：静态 exact 情形

若：

$$
\delta_t=0,
\qquad
 e_{k,t}^{\mathrm{sk}}=0,
\qquad
 b_t=0,
\qquad
 r_t=0,
$$

且 $\eta_t,\xi_t=\Theta(T^{-1/2})$，则：

$$
\boxed{
\mathbb E\|\nabla f(\theta_R)\|^2
=
O(T^{-1/2})
+
O\left(
\frac{1}{T}
\sum_{t=1}^{T}
\Gamma_t^a
\sum_{k=1}^{K}
\frac{
\omega_ke_{k,t}^{\mathrm{stat}}
}{
\Delta_{\mathrm{flag},t}
}
\right)
}
$$

若 EMA covariance error 进一步满足：

$$
e_{k,t}^{\mathrm{stat}}
=
O\left(
N_{\mathrm{eff},k}^{-1/2}
\right),
$$

则：

$$
\boxed{
\mathbb E\|\nabla f(\theta_R)\|^2
=
O(T^{-1/2})
+
O\left(
\sum_{k=1}^{K}
\frac{
\overline{\Gamma^a}\omega_k
}{
\Delta_{\mathrm{flag},k}
\sqrt{N_{\mathrm{eff},k}}
}
\right)
}
$$

### Corollary 18.2：漂移稳态

常数 $\xi$ 下：

$$
\boxed{
\limsup_t e_t^{\mathrm{track}}
=
O\left(
\frac{e^{\mathrm{stat}}+e^{\mathrm{sk}}+\|b\|}{\mu}
+
\frac{\xi\sigma_{\mathrm{flag}}^2}{\mu}
+
\frac{\delta}{\mu\xi}
\right)
}
$$

因此参数 stationarity 的额外项为：

$$
\boxed{
O\left(
\Gamma^a
\left[
\frac{e^{\mathrm{stat}}+e^{\mathrm{sk}}+\|b\|}{\mu}
+
\frac{\xi\sigma_{\mathrm{flag}}^2}{\mu}
+
\frac{\delta}{\mu\xi}
\right]
\right)
}
$$

### Corollary 18.3：单一更新系数退化

若所有层级和残差使用相同系数：

$$
a_{1,t}=\cdots=a_{K,t}=a_{\perp,t},
$$

则：

$$
\Gamma_t^a=0,
$$

所有 flag-induced preconditioner error 项消失。此时层级空间仍可用于诊断，但不会改变线性参数更新。这说明 STRATA 的优化收益必须来自“嵌套空间”和“层级更新规则”共同作用，而不能只来自保存一组旗基。

---

## 4.8 理论量与实验量的一一对应

| 理论量 | 实验估计 |
|---|---|
| $e_{k,t}^{\mathrm{stat}}$ | 大窗口 covariance 与在线 EMA covariance 的谱范数差 |
| $e_{k,t}^{\mathrm{sk}}$ | full covariance 与 sketched covariance 的谱范数差 |
| $\Delta_{\mathrm{flag},t}$ | 累计矩阵 $A_{j,t}$ 的相关 eigengap |
| $\mu_{\mathrm{flag},t}$ | 最优旗邻域 Riemannian Hessian 最小曲率 proxy |
| $\sigma_{\mathrm{flag},t}^2$ | 多 minibatch flag gradient 方差 |
| $\delta_t$ | 相邻 oracle flag 的 principal-angle distance |
| $e_t^{\mathrm{track}}$ | 在线旗与离线局部最优旗的距离 |
| $\Gamma_t^a$ | 相邻层级更新系数差绝对值之和 |
| $\|B_t-B_t^{\star}\|_2$ | 实际旗与 oracle flag 构造的预条件器差 |
| $\|r_t\|$ | transport、状态旋转、clipping 和数值近似的更新残差 |

实验必须至少检验两个定量关系：

$$
\boxed{
 e_t^{\mathrm{track}}
\propto
\frac{e_t^{\mathrm{stat}}+e_t^{\mathrm{sk}}}{\Delta_{\mathrm{flag},t}}
+
\frac{\xi_t\sigma_{\mathrm{flag},t}^2}{\mu_t}
+
\frac{\delta_t}{\mu_t\xi_t}
}
$$

以及：

$$
\boxed{
\text{optimization degradation}
\propto
\Gamma_t^a
\times
 e_t^{\mathrm{track}}
}
$$

这两个关系是理论是否被数据支持的主要判据。


# 5. 实验总协议

## 5.1 Seed 协议

所有实验只运行：

$$
\boxed{\textbf{1 seed}}
$$

固定 seed：

$$
\texttt{seed}=42.
$$

不运行额外 seed，不报告均值和标准差。原因不是统计优势，而是本项目优先进行广任务、长训练和完整动力学记录。论文中必须明确说明：

> All experiments use a single fixed seed. We report complete long-horizon trajectories rather than multi-seed averages.

不能把单 seed 结果写成高置信度统计结论。

## 5.2 长训练协议

所有正式长训练必须满足：

$$
\boxed{\textbf{epochs}\ge1000}
$$

具体规定：

- 小型视觉、表格、图学习、持续学习：至少 1000 epochs；
- 中型视觉：至少 1000 epochs；
- PINN 和 operator learning：至少 1000 epochs；
- 语言模型按完整 token 预算训练，并将评估点数量设置为不少于 1000 个 epoch-equivalent checkpoints；
- 不因验证集暂时下降提前停止；
- 所有方法使用相同训练长度；
- 允许保存 best validation checkpoint，但训练必须继续完成。

## 5.3 全量结果保留

每个训练 run 必须保存：

1. 每个 epoch 的 train loss；
2. 每个 epoch 的 validation metric；
3. 每个 epoch 的 test metric，仅用于最终画图，不参与选择；
4. 每个 epoch 的学习率；
5. 每个 epoch 的梯度范数；
6. 每个 epoch 的各层级能量；
7. 每个 flag update 的投影矩阵或压缩表示；
8. 每个 flag update 的 principal angles；
9. 每个 flag update 的 nesting gap；
10. 每个 flag update 的 reconstruction error；
11. 每个 epoch 的 wall-clock；
12. 每个 epoch 的 peak memory；
13. 每 10 epochs 一个 checkpoint；
14. epoch 0、1、2、5、10、20、50、100、200、500、1000 的永久 checkpoint；
15. 所有异常、NaN、重新加载和运行中断日志。

任何不利结果都不能删除。

## 5.4 配置选择

所有方法只通过验证集选择超参数。每个配置仍只运行 1 seed。

测试集不能用于：

- 选择任务；
- 选择 rank；
- 选择层级数；
- 选择学习率；
- 选择 flag update frequency；
- 选择主文结果。

主文可展示验证集预先选出的效果最好任务族，但完整矩阵必须公开。

---

# 6. 实验任务与清晰表格

## 6.1 语言模型预训练

### 固定设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epoch-equivalent evaluations | $\ge1000$ |
| Checkpoint interval | 每 10 个 evaluation units |
| Optimizers | AdamW, GaLore, Single-Grassmann, Independent-Grassmann, STRATA |
| Rank budgets | 16, 32, 64, 128 |
| Flag levels | 3 |
| Models | GPT-2 Small, GPT-2 Medium, 125M decoder |
| Datasets | WikiText-103, OpenWebText subset, C4 subset |

### 结果表

| Dataset | Model | Method | Rank | Final val PPL | Best val PPL | Final train loss | Tokens to best | Peak GB | Tokens/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| WikiText-103 | GPT-2 Small | AdamW | Full | TBD | TBD | TBD | TBD | TBD | TBD |
| WikiText-103 | GPT-2 Small | GaLore | 64 | TBD | TBD | TBD | TBD | TBD | TBD |
| WikiText-103 | GPT-2 Small | Single-Grassmann | 64 | TBD | TBD | TBD | TBD | TBD | TBD |
| WikiText-103 | GPT-2 Small | STRATA $(16,32,64)$ | 64 | TBD | TBD | TBD | TBD | TBD | TBD |

### 必须绘图

- train loss vs tokens；
- validation PPL vs tokens；
- level energy vs tokens；
- principal angle vs tokens；
- nesting gap vs tokens；
- projection error vs tokens；
- memory vs validation PPL；
- wall-clock vs validation PPL。

---

## 6.2 视觉分类

### 固定设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 1000 |
| Evaluation | 每 1 epoch |
| Checkpoint | 每 10 epochs |
| Datasets | CIFAR-10, CIFAR-100, Tiny-ImageNet, ImageNet-100 |
| Models | ResNet-18, ResNet-50, ViT-S/16, ConvNeXt-T |
| Methods | SGD-M, AdamW, Muon, GaLore, Single-Grassmann, STRATA |

### 结果表

| Dataset | Model | Method | Rank/flag | Final top-1 | Best top-1 | Final train loss | Epoch to 90% best | Peak GB | sec/epoch |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | ResNet-18 | AdamW | Full | TBD | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 | ResNet-18 | GaLore | 64 | TBD | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 | ResNet-18 | Single-Grassmann | 64 | TBD | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 | ResNet-18 | STRATA | $(16,32,64)$ | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 6.3 持续学习

### 固定设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Total epochs | $\ge1000$ |
| Task schedule | 固定顺序 |
| Datasets | Split CIFAR-100, Rotated MNIST, DomainNet, CORe50 |
| Backbones | MLP, ResNet-18 |
| Base methods | Fine-tuning, replay, EWC |
| Optimizers | AdamW, GaLore, Single-Grassmann, STRATA |

### 结果表

| Dataset | Base CL method | Optimizer | Final avg acc | Final forgetting | Forward transfer | Mean recovery epochs | Inner overlap | Outer overlap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Split CIFAR-100 | Replay | AdamW | TBD | TBD | TBD | TBD | TBD | TBD |
| Split CIFAR-100 | Replay | GaLore | TBD | TBD | TBD | TBD | TBD | TBD |
| Split CIFAR-100 | Replay | Single-Grassmann | TBD | TBD | TBD | TBD | TBD | TBD |
| Split CIFAR-100 | Replay | STRATA | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 6.4 人工非平稳训练

### Shift 设置

| Shift ID | Dataset | Shift type | Shift epochs | Total epochs |
|---|---|---|---|---:|
| S1 | CIFAR-10 | label distribution | 250, 500, 750 | 1000 |
| S2 | CIFAR-100 | class subset switch | 250, 500, 750 | 1000 |
| S3 | Rotated MNIST | angle sequence | 每 100 epochs | 1000 |
| S4 | WikiText-103 | corpus segment switch | 25%, 50%, 75% tokens | full budget |

### 结果表

| Shift ID | Method | Pre-shift loss | Peak loss | Recovery epochs | Post-recovery loss | Inner drift | Middle drift | Outer drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | AdamW | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| S1 | Single-Grassmann | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| S1 | STRATA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 6.5 图学习

### 固定设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 1000 |
| Datasets | Cora, Citeseer, PubMed, ogbn-arxiv |
| Models | GCN, GAT, GraphSAGE, GCNII |
| Evaluation | 每 epoch |
| Full logs | 必须 |

### 结果表

| Dataset | Model | Method | Final val | Final test | Best val | Epoch to best | Peak GB | Mean drift |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | GraphSAGE | AdamW | TBD | TBD | TBD | TBD | TBD | TBD |
| ogbn-arxiv | GraphSAGE | GaLore | TBD | TBD | TBD | TBD | TBD | TBD |
| ogbn-arxiv | GraphSAGE | Single-Grassmann | TBD | TBD | TBD | TBD | TBD | TBD |
| ogbn-arxiv | GraphSAGE | STRATA | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 6.6 科学机器学习

### 固定设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 2000 |
| Tasks | Poisson, Burgers, Darcy, FNO |
| Metrics | Relative $L_2$, PDE residual, frequency error |
| Evaluation | 每 epoch |

### 结果表

| Task | Model | Method | Final rel. $L_2$ | Best rel. $L_2$ | PDE residual | Low-freq error | High-freq error | Peak GB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Burgers | PINN | AdamW | TBD | TBD | TBD | TBD | TBD | TBD |
| Burgers | PINN | Single-Grassmann | TBD | TBD | TBD | TBD | TBD | TBD |
| Burgers | PINN | STRATA | TBD | TBD | TBD | TBD | TBD | TBD |

---

# 7. 理论对应实验

## 7.1 空间最优性

| Dataset | Layer | Independent optimum error | Best flag error | Excess flag error | Nesting gap | Shared-basis commutator |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-100 | block 1 | TBD | TBD | TBD | TBD | TBD |
| WikiText-103 | MLP layer 1 | TBD | TBD | TBD | TBD | TBD |

交换性 proxy：

$$
\operatorname{Comm}(i,j)
=
\frac{
\|C_iC_j-C_jC_i\|_F
}{
\|C_i\|_F\|C_j\|_F+\epsilon
}.
$$

## 7.2 Landscape

对小型可完整计算任务：

- 线性回归；
- 矩阵分解；
- 小型 MLP；
- synthetic covariance。

记录：

| Setting | Number of starts | Global optimum reached | Saddle escape steps | Hessian min eig. | Objective gap |
|---|---:|---:|---:|---:|---:|
| Commuting covariance | 100 initializations | TBD | TBD | TBD | TBD |
| Near-commuting covariance | 100 initializations | TBD | TBD | TBD | TBD |

这些是理论诊断，可运行多个初始化，但每个深度学习正式实验仍只使用 seed 42。

## 7.3 Convergence

| Dataset | Method | Final gradient norm | Min gradient norm | Loss AUC 0–1000 | Final loss | Bound proxy |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-100 | AdamW | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 | Single-Grassmann | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 | STRATA | TBD | TBD | TBD | TBD | TBD |

## 7.4 Error bound

| Layer | Timescale | Eigengap | Covariance error | DK bound | Observed angle | Sketch error |
|---|---|---:|---:|---:|---:|---:|
| Layer 1 | Long | TBD | TBD | TBD | TBD | TBD |
| Layer 1 | Medium | TBD | TBD | TBD | TBD | TBD |
| Layer 1 | Short | TBD | TBD | TBD | TBD | TBD |

## 7.5 Generalization

| Dataset | Method | Train metric | Test metric | Gap | Stability proxy | Update-noise bound | Effective dimension |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | AdamW | TBD | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 | Single-Grassmann | TBD | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 | STRATA | TBD | TBD | TBD | TBD | TBD | TBD |

---

# 8. 必要消融

| ID | 消融 | 目的 |
|---|---|---|
| A1 | $K=1$ Single-Grassmann | 检验旗层级是否必要 |
| A2 | 多个独立 Grassmann，总秩相同 | 检验嵌套约束是否必要 |
| A3 | 所有 $\rho_k$ 相同 | 检验多时间尺度统计 |
| A4 | 所有 $\beta_k$ 相同 | 检验层级记忆 |
| A5 | 所有 $a_k$ 相同 | 检验层级学习率 |
| A6 | 固定旗 | 检验在线流形更新 |
| A7 | 周期 SVD | 比较连续旗更新与离散刷新 |
| A8 | 随机嵌套旗 | 检验数据驱动空间 |
| A9 | 打乱层级顺序 | 检验长期—短期解释 |
| A10 | 不做 transport | 检验状态一致性 |
| A11 | 无残差 | 检验最大秩覆盖不足 |
| A12 | full-rank base optimizer | 检验性能—内存边界 |

### 消融结果表

| Ablation | Final task metric | Final train loss | Long error | Short error | Nesting gap | Drift | Peak GB | sec/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full STRATA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A6 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

所有消融运行 1000 epochs，seed 固定为 42，完整保留所有 epoch 日志。

---

# 9. 超参数

| 类别 | 搜索范围 |
|---|---|
| Base learning rate | $\{10^{-5},3\cdot10^{-5},10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3},10^{-2}\}$ |
| Maximum rank $R$ | $\{8,16,32,64,128\}$ |
| Number of levels $K$ | $\{2,3,4\}$ |
| Rank template | $(R/4,R/2,R)$, $(R/8,R/2,R)$, $(R/2,3R/4,R)$ |
| Long $\rho_1$ | $\{0.99,0.995,0.999\}$ |
| Mid $\rho_2$ | $\{0.95,0.98,0.99\}$ |
| Short $\rho_3$ | $\{0.8,0.9,0.95\}$ |
| Long $\beta_1$ | $\{0.99,0.995,0.999\}$ |
| Mid $\beta_2$ | $\{0.9,0.95,0.98\}$ |
| Short $\beta_3$ | $\{0,0.8,0.9\}$ |
| Flag learning rate $\xi$ | $\{10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3},10^{-2}\}$ |
| Flag update interval | $\{10,25,50,100,200\}$ |
| Oversampling $p$ | $\{4,8,16\}$ |

每个方法的调参总 run 数必须相同。所有 trial 仍只运行 seed 42。

---

# 10. 必须通过的数值测试

## 10.1 正交性

$$
\|U^\top U-I\|_F<10^{-5}.
$$

## 10.2 嵌套性

$$
\|(I-P_{k+1})P_k\|_F<10^{-5}.
$$

## 10.3 分解正确性

$$
\left\|
G
-
\sum_kG\Pi_k
-
G\Pi_\perp
\right\|_F
<
10^{-6}\|G\|_F.
$$

## 10.4 能量守恒

$$
\left|
\|G\|_F^2
-
\sum_k\|G\Pi_k\|_F^2
-
\|G\Pi_\perp\|_F^2
\right|
<
10^{-6}\|G\|_F^2.
$$

## 10.5 Gauge invariance

块内旋转：

$$
U^{[k]}\mapsto U^{[k]}O_k
$$

不能改变所有 $P_j$、目标值和重构梯度。

## 10.6 水平投影

水平梯度必须与垂直空间正交。

## 10.7 Retraction

retraction 后：

$$
U_{\mathrm{new}}^\top U_{\mathrm{new}}=I.
$$

## 10.8 Transport

当：

$$
U_{\mathrm{new}}=U
$$

时，transport 必须退化为恒等变换。

## 10.9 单层退化

当：

$$
K=1
$$

时，STRATA 必须退化为配对 Single-Grassmann optimizer。

## 10.10 全秩退化

当：

$$
R=d
$$

时：

$$
\Pi_\perp=0.
$$

## 10.11 有限差分

旗目标的 Riemannian directional derivative 必须与有限差分一致。

## 10.12 理论 synthetic test

构造共享特征基协方差，验证：

- 全局最优旗与 Theorem 1 一致；
- 非最优临界点有负曲率方向；
- 误差随 $\varepsilon/\Delta$ 缩放；
- tracking error 随 drift 增大。

---

# 11. 成功与失败标准

## 11.1 成功标准

至少满足：

1. 在一个主要任务族上，STRATA 优于相同最大秩的 Single-Grassmann；
2. 在至少两个模型上方向一致；
3. 至少一个 1000-epoch 中型任务成立；
4. 多时间尺度重构误差低于单一空间；
5. 内层漂移低于外层；
6. 任务切换时外层先响应，内层较稳定；
7. 独立子空间不能在相同代价下完全复现结果；
8. 理论 bound 与观测趋势一致；
9. 峰值显存低于 full AdamW；
10. 所有完整日志和失败任务公开保留。

## 11.2 失败标准

若出现以下情况，必须降低结论：

1. 嵌套缺口长期很大；
2. 独立子空间稳定优于旗结构；
3. 中间层能量接近零；
4. 所有层漂移速度相同；
5. 1000 epochs 后优势消失；
6. 结果只来自早期训练；
7. transport 造成明显不稳定；
8. flag update 时间抵消显存收益；
9. 单 seed 结果对超参数极度敏感；
10. 理论假设与实测协方差结构不符。

---

# 12. 推荐论文叙述

## Observation

低秩梯度方法通常使用单一子空间，但长训练记录显示，不同时间尺度的梯度统计共享一个稳定核心，同时需要逐层增加快速方向。

## Limitation

单一 Grassmann 子空间必须在稳定性和响应速度之间折中。多个独立子空间又会重叠，不能保证不同维数表示一致。

## Method

STRATA 将优化器状态放在旗流形上，在线学习长期、中期和短期嵌套梯度子空间，并为不同层级配置独立动量、学习率和低维自适应状态。

## Theory

- 给出共享特征基下的全局最优旗；
- 给出非交换协方差下的近似最优误差；
- 证明共享基条件下无伪局部极大值和 strict-saddle landscape；
- 给出非凸 convergence 和 PL 条件下的线性收敛；
- 给出 covariance、sketch、flag estimation 和 drifting tracking error bounds；
- 给出基于稳定性和更新噪声的 generalization bounds。

## Experiments

所有正式实验只使用 seed 42。所有长训练至少 1000 epochs。保存每个 epoch 和每个 flag update 的完整结果，主文展示长时间曲线而不是只展示最终点。

---

# 13. 推荐标题

> **STRATA: Multi-Scale Manifold Optimization with Nested Gradient Subspaces**

候选标题：

> **STRATA: Hierarchical Gradient Optimization on Flag Manifolds**

> **Nested Geometry for Multi-Scale Gradient Optimization**

---

# References

1. Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., and Tian, Y. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*. ICML 2024.
2. Su, D., Gu, A., Xu, J., Tian, Y., and Zhao, J. *GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection*. arXiv:2504.20437, 2025.
3. Ye, K., and Wong, K. W. *Optimization on Flag Manifolds*. Mathematical Programming, 2022.
4. Lai, Z., Lim, L.-H., and Ye, K. *Simpler Flag Optimization*. arXiv:2212.00212, 2022.
5. Szwagier, T., and Pennec, X. *Nested Subspace Learning with Flags*. arXiv:2502.06022, 2025.
6. Jin, D. et al. *Online Subspace Learning on Flag Manifolds for System Identification*. arXiv:2511.06416, 2025.
