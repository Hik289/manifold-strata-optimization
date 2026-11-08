# How STRATA-AdamW works

This guide explains the current implementation in
[`strata_adamw.py`](../experiments/strata_pkg/strata_adamw.py) and
[`geometry.py`](../experiments/strata_pkg/geometry.py). The
[API guide](API.md) lists the options. The paper source contains the full
[convergence](../paper/sections/theory/05_convergence.tex) and
[generalization](../paper/sections/theory/06_generalization.tex) analyses.

## 1. The basic idea

A gradient matrix can contain directions that persist for many steps and
others that change quickly. STRATA tracks gradient correlations with several
memory lengths and uses a set of nested subspaces to organize its updates.

A **subspace** is a collection of directions closed under linear combinations.
A **basis** is a matrix whose columns describe those directions. **Nested**
means each smaller subspace is contained in the next one. A sequence of nested
subspaces is called a **flag**:

$$
\mathcal U_1\subset\mathcal U_2\subset\cdots\subset\mathcal U_K.
$$

The basis belongs to the optimizer's stored state. Model weights remain
unconstrained, and the forward pass uses the original model.

## 2. Matrices, subspaces and blocks

Write the working gradient as $G\in\mathbb R^{m\times d}$, with $d\le m$.
The implementation transposes a matrix when needed so that the subspaces lie
along its smaller dimension. A convolution kernel is first reshaped into a
matrix.

Let the requested subspace sizes be

$$
0=r_0<r_1<\cdots<r_K=R<d.
$$

STRATA stores one basis $U\in\mathbb R^{d\times R}$ with orthonormal columns:
$U^\top U=I_R$. Its first $r_k$ columns span $\mathcal U_k$. Because all levels
use prefixes of the same matrix, they remain nested.

Split the columns into separate blocks:

$$
U=[U^{[1]},\ldots,U^{[K]}],\qquad
U^{[k]}\in\mathbb R^{d\times q_k},\qquad q_k=r_k-r_{k-1}.
$$

For example, `ranks=[16,32,64]` gives subspace sizes 16, 32 and 64, but block
sizes **16, 16 and 32**. The subspaces overlap by design; the separate blocks
are mutually orthogonal.

## 3. What the optimizer stores

Each eligible parameter matrix has its own state:

| State | Purpose |
| --- | --- |
| $U$ | Current orthonormal basis for the nested subspaces. |
| $M_k$ | Running average of gradient coordinates in block $k$. |
| $V_k$ | Running average of squared gradient coordinates in block $k$. |
| $c_{M,k},c_{V,k}$ | Separate step counts for correcting the two averages' initial bias. |
| $\Omega_k$ | Fixed random probe matrix used to compress covariance information. |
| $Y_k$ | Running compressed covariance estimate for memory level $k$. |

The basis and probes are initialized from seeded random matrices followed by
QR factorization. The averages start at zero. The probe width is
$s_k=\min(d,r_k+p)$, where `p` is the requested number of extra columns.

The following sections describe one optimizer step. The gradient coordinates
use the current basis. Any basis refresh affects the next step's coordinates.

## 4. Track gradient history

The matrix $G^\top G$ measures correlations between gradient columns. Storing
it explicitly costs $d^2$ entries. STRATA instead stores its action on the
smaller probe matrix $\Omega_k\in\mathbb R^{d\times s_k}$:

$$
Y_k\leftarrow\rho_kY_k+(1-\rho_k)G^\top G\Omega_k.
$$

This is an exponential moving average. A larger $\rho_k$ retains information
for longer. The code computes $G^\top(G\Omega_k)$ without forming $G^\top G$.
The compact matrix $Y_k$ is called a **covariance sketch**.

For basis updates, the sketch defines the regularized Nyström approximation

$$
\widetilde C_k=Y_kB_k^{-1}Y_k^\top,\qquad
B_k=\operatorname{sym}(\Omega_k^\top Y_k)+\gamma I,
$$

where $\operatorname{sym}(A)=(A+A^\top)/2$ and $\gamma>0$ is the regularizer.
Nyström is the name of this low-rank matrix approximation. In training, the
code applies $\widetilde C_k$ to a matrix by solving the small system involving
$B_k$; it does not build the full $d\times d$ approximation.

## 5. Split the gradient

Project the gradient onto each basis block:

$$
Z_k=GU^{[k]}.
$$

The coordinates $Z_k$ have shape $m\times q_k$. Mapping them back gives the
corresponding component of the original gradient. The part outside the largest
subspace is the **residual**:

$$
G_\perp=G-GUU^\top,\qquad
G=\sum_{k=1}^{K}Z_kU^{[k]\top}+G_\perp.
$$

Because the blocks are orthogonal, their squared Frobenius norms add up:

$$
\lVert G\rVert_F^2=
\sum_{k=1}^{K}\lVert Z_k\rVert_F^2+\lVert G_\perp\rVert_F^2.
$$

The numerical tests check both reconstruction and this energy identity.

## 6. Apply Adam within each block

For every block, update the running gradient and squared-gradient averages:

$$
M_k\leftarrow\beta_{1,k}M_k+(1-\beta_{1,k})Z_k,
\qquad
V_k\leftarrow\beta_{2,k}V_k+(1-\beta_{2,k})(Z_k\odot Z_k).
$$

Increase both step counts by one and apply Adam's bias correction:

$$
\widehat M_k=\frac{M_k}{1-\beta_{1,k}^{c_{M,k}}},\qquad
\widehat V_k=\frac{V_k}{1-\beta_{2,k}^{c_{V,k}}},\qquad
H_k=\frac{\widehat M_k}{\sqrt{\widehat V_k}+\epsilon}.
$$

The product, square root and division in these formulas act entry by entry.
The positive constant $\epsilon$ prevents division by very small values; the
implementation defaults to $10^{-6}$.

Map the block updates back to the weight matrix and add the normalized
residual:

$$
D=\sum_{k=1}^{K}a_kH_kU^{[k]\top}
+a_\perp\frac{G_\perp}{\lVert G_\perp\rVert_F/\sqrt{md}+\epsilon}.
$$

The coefficients $a_k$ and $a_\perp$ control the contributions of the separate
blocks and the residual. The residual has no stored Adam moments.

With learning rate $\eta$ and weight decay $\lambda$, the weight update is

$$
W\leftarrow(1-\eta\lambda)W-\eta D.
$$

If the gradient was transposed or reshaped, $D$ is returned to the parameter's
original shape before this update.

## 7. Update the nested subspaces

STRATA adjusts the basis to increase the gradient energy captured at each
memory level. For fixed current covariance estimates, its objective is

$$
\Phi(U)=\sum_{k=1}^{K}\omega_k
\operatorname{tr}\left(U_{:r_k}^{\top}\widetilde C_kU_{:r_k}\right).
$$

Here $U_{:r_k}$ denotes the first $r_k$ columns. Each term measures the
estimated energy inside that subspace; $\omega_k$ sets its weight.

Define $A_j=\sum_{k=j}^{K}\omega_k\widetilde C_k$. The ordinary matrix gradient
of this objective is

$$
H=[2A_1U^{[1]},\ldots,2A_KU^{[K]}].
$$

An unrestricted step could destroy orthogonality. STRATA first removes the
component that points away from the orthogonality constraint:

$$
H_{\mathrm{st}}=H-U\operatorname{sym}(U^\top H).
$$

It then removes rotations within each existing block, which change coordinates
without changing the represented subspaces:

$$
H_{\mathrm{hor}}=H_{\mathrm{st}}
-U\operatorname{blkdiag}(U^\top H_{\mathrm{st}}).
$$

`blkdiag` keeps the diagonal blocks of sizes $q_1,\ldots,q_K$ and sets the
other entries to zero. The result is the **horizontal gradient** used by the
implementation.

Take a step of size $\xi$ and restore orthonormal columns with QR factorization:

$$
\bar U=\operatorname{qf}(U+\xi H_{\mathrm{hor}}).
$$

The function `qf` returns the orthonormal factor, with a fixed sign convention.
This operation is called a **QR retraction**. The code refreshes the basis on
the first matrix update and then every `tau` matrix updates. The direction
$D$ has already been computed, so the refreshed basis is used on the next step.

## 8. Align coordinates and update the stored averages

The columns returned by QR can rotate within a block. STRATA aligns each new
block with the old one using a small singular value decomposition:

$$
\bar U^{[k]\top}U^{[k]}=L_k\Sigma_kR_k^\top,\qquad
O_k=L_kR_k^\top,\qquad
U_{\mathrm{new}}^{[k]}=\bar U^{[k]}O_k.
$$

This is **orthogonal Procrustes alignment**: choose the rotation that makes
the new block as close as possible to the old block in Frobenius norm.

The implementation then applies

$$
M_k\leftarrow M_kO_k^\top,\qquad
V_k\leftarrow0,\qquad c_{V,k}\leftarrow0,
$$

while keeping $c_{M,k}$ unchanged. Thus the gradient average keeps its history,
and the squared-gradient average starts again in the updated coordinates.
Separate bias-correction counts are necessary because only one average resets.
The paper's [adaptive-state analysis](../paper/sections/theory/04_adaptive_state.tex)
examines the errors caused by changing bases and transferring stored state.

## 9. Parameters that use dense AdamW

The subspace branch is used only when a parameter is a matrix or a 4D
convolution kernel and its smaller matrix dimension exceeds $R$. All other
parameters use the dense AdamW branch, including biases, normalization
parameters and matrices that are too small for the requested rank.

This choice avoids trying to fit an oversized basis into a small parameter.
Each parameter with no gradient is skipped for that optimizer step.

## 10. Reading the theory and checking the implementation

The full proofs remain in the paper source. These files provide the relevant
starting points:

| Question | Paper source |
| --- | --- |
| What geometry does the basis update use? | [Flag geometry](../paper/sections/theory/01_geometry.tex) |
| How do covariance estimates and changing data affect the subspaces? | [Statistics and tracking](../paper/sections/theory/03_statistics_tracking.tex) |
| How does changing the basis affect Adam state? | [Adaptive state](../paper/sections/theory/04_adaptive_state.tex) |
| Under what assumptions do the updates converge? | [Convergence](../paper/sections/theory/05_convergence.tex), [implemented Adam dynamics](../paper/sections/theory/09_actual_adam_dynamics.tex) |
| How does the analysis bound generalization error? | [Stability](../paper/sections/theory/06_generalization.tex), [PAC-Bayes bounds](../paper/sections/theory/08_pac_bayes.tex), [direction-dependent bounds](../paper/sections/theory/10_directional_generalization.tex) |

Each theorem states its assumptions. The bounds do not imply that STRATA
outperforms every baseline on every task; empirical comparisons use the
separate rules in the [reproduction guide](REPRODUCIBILITY.md).

Run the numerical suite from the repository root:

```bash
python experiments/test_numerical.py
```

Its 13 checks cover orthogonality, nesting, gradient reconstruction, energy
conservation, basis rotations, QR retraction, stored-state updates and the
one-level comparison with Single-Grassmann. Under matched settings, the
one-level implementation follows the Single-Grassmann update. Synthetic checks
and finite differences also test several quantities used in the analysis.
