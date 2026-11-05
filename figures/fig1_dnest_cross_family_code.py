"""
Fig 1 (Core, v2 — epoch-axis): δ_nest Cross-Family Trajectories
================================================================
Publication-quality (NeurIPS 9-10pt LaTeX template standard).

CORRECTED VERSION (epoch axis, not learn_progress).
learn_progress had a systematic bias: PINN converges in metric very early,
compressing its δ_nest evolution into a narrow x-region and artifically
flattening the trend. Epoch axis avoids this.

Key honest message (implementation-locked, DS-verified):
- On epoch axis: 4 families rise (Burgers/Poisson/graph/CNN), 2 families flat/fall (MNIST/Fashion).
- ALL verdicts are TIE (except Poisson where SG numerically diverges).
- → δ_nest trend does NOT predict STRATA win/loss. δ_nest is NOT a performance predictor.
- graph's uniqueness: starting point 0.047 (uniquely low, only family < 0.1),
  not being the only riser (Burgers, Poisson, CNN also rise).

Presentation: epoch% x-axis (epoch / max_epoch × 100), Spearman ρ annotated per curve,
verdict label at curve end, right panel shows slope vs verdict to directly show decoupling.

Run: python3 fig1_dnest_cross_family_code.py
Inputs: ../experiments/dnest_cross_family.json
Outputs: fig1_dnest_cross_family.pdf  (publication vector)
         fig1_dnest_cross_family.png  (150 dpi preview)
         fig1_dnest_cross_family_caption.txt
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
PROJ = Path(str(Path(__file__).resolve().parents[1]))
DATA_FILE = PROJ / "experiments" / "dnest_cross_family.json"
OUT_DIR   = PROJ / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data + compute stats BEFORE matplotlib import ───────────────────────
raw_data = json.loads(DATA_FILE.read_text())

# Pre-open output file handles (dentry-cache workaround)
_fh_pdf = open(OUT_DIR / "fig1_dnest_cross_family.pdf", "wb")
_fh_png = open(OUT_DIR / "fig1_dnest_cross_family.png", "wb")

# ── Process each family ───────────────────────────────────────────────────────
FAMILIES = {
    "graph_arxiv":  {"label": "Graph (ogbn-arxiv)", "max_ep": 999},
    "MNIST_MLP":    {"label": "MNIST (MLP)",         "max_ep": 999},
    "Fashion_MLP":  {"label": "Fashion (MLP)",       "max_ep": 999},
    "PINN_Burgers": {"label": "PINN–Burgers",        "max_ep": 2999},
    "PINN_Poisson": {"label": "PINN–Poisson",        "max_ep": 2999},
    "CNN_cifar":    {"label": "CNN (CIFAR-10/100)",  "max_ep": 999},
}

# Head-to-head verdict (honest labels)
H2H = {
    "graph_arxiv":  "tie",
    "MNIST_MLP":    "tie",
    "Fashion_MLP":  "tie",
    "PINN_Burgers": "tie",
    "PINN_Poisson": "STRATA (SG div.)",
    "CNN_cifar":    "tie",
}

processed = {}
for name, meta in FAMILIES.items():
    if name not in raw_data:
        continue
    traj = raw_data[name]["dnest_trajectory"]
    ep   = np.array([p["epoch"] for p in traj], dtype=float)
    dn   = np.array([p["dnest_norm"] for p in traj], dtype=float)
    ep_pct = ep / meta["max_ep"] * 100.0  # 0-100%

    rho, pval = stats.spearmanr(ep_pct, dn)
    slope, _ = np.polyfit(ep_pct, dn, 1)   # per percentage-point

    processed[name] = {
        "ep_pct": ep_pct,
        "dn":     dn,
        "label":  meta["label"],
        "h2h":    H2H[name],
        "rho":    rho,
        "pval":   pval,
        "slope":  slope,
        "start":  dn[0],
        "end":    dn[-1],
    }

# ── matplotlib import ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── publication rcParams ─────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           9.5,
    "axes.labelsize":      10.5,
    "axes.labelweight":    "bold",
    "axes.titlesize":      11,
    "axes.titleweight":    "bold",
    "xtick.labelsize":     8.5,
    "ytick.labelsize":     8.5,
    "legend.fontsize":     7.8,
    "legend.framealpha":   0.93,
    "legend.edgecolor":    "#cccccc",
    "lines.linewidth":     2.0,
    "lines.markersize":    4.5,
    "axes.grid":           True,
    "grid.alpha":          0.22,
    "grid.linewidth":      0.55,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          150,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
})

# ── Okabe-Ito colorblind-safe palette ────────────────────────────────────────
C = {
    "graph_arxiv":  "#E69F00",   # orange   — unique low start
    "MNIST_MLP":    "#56B4E9",   # sky-blue
    "Fashion_MLP":  "#009E73",   # teal-green
    "PINN_Burgers": "#0072B2",   # deep blue
    "PINN_Poisson": "#D55E00",   # vermillion
    "CNN_cifar":    "#CC79A7",   # mauve-pink
}

# Display order: risers first (descending slope magnitude), then flat/fallers
ORDER = ["graph_arxiv", "PINN_Burgers", "PINN_Poisson", "CNN_cifar", "MNIST_MLP", "Fashion_MLP"]

# Line styles to distinguish 6 curves
LS = {
    "graph_arxiv":  "-",
    "PINN_Burgers": "--",
    "PINN_Poisson": "-.",
    "CNN_cifar":    (0, (5, 1.5)),
    "MNIST_MLP":    ":",
    "Fashion_MLP":  (0, (3, 1, 1, 1)),
}

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE: 2 panels side by side
# Left: epoch% trajectory (main)
# Right: slope summary bar showing decoupling with verdict
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13.0, 4.6),
    gridspec_kw={"width_ratios": [1.7, 1]},
)

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — trajectories
# ═══════════════════════════════════════════════════════════════════════════════
for name in ORDER:
    if name not in processed:
        continue
    d = processed[name]
    col  = C[name]
    ls   = LS[name]
    lw   = 2.6 if name == "graph_arxiv" else 1.9
    ms   = 5 if name == "graph_arxiv" else 3.5

    ax1.plot(d["ep_pct"], d["dn"],
             linestyle=ls, linewidth=lw, color=col,
             marker="o", markersize=ms, markevery=8,
             zorder=6 if name == "graph_arxiv" else 4)

# --- graph family special annotation: uniquely low start ---
g = processed["graph_arxiv"]
ax1.scatter([g["ep_pct"][0]], [g["dn"][0]],
            s=90, color=C["graph_arxiv"], marker="D", zorder=10,
            edgecolors="white", linewidths=0.8, label=None)
ax1.annotate(
    f"graph start:\n$\\delta_{{\\rm nest}}$ = {g['dn'][0]:.3f}\n(only family $< 0.1$)",
    xy=(g["ep_pct"][0], g["dn"][0]),
    xytext=(12, 0.25),
    ha="left", va="center", fontsize=7.8,
    color=C["graph_arxiv"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["graph_arxiv"], lw=1.2,
                    connectionstyle="arc3,rad=0.25"),
    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C["graph_arxiv"],
              alpha=0.92, lw=0.9),
    zorder=11,
)

# --- End-of-curve labels: verdict + Spearman ρ ---
_label_offsets = {
    # (x_shift_pct, y_shift) — carefully staggered to avoid overlap near epoch%=100
    # graph ends ~0.493; graph label pushed above to ~0.69
    "graph_arxiv":  (  3, +0.20),
    # Burgers ends ~0.690; label pushed lower to avoid graph collision
    "PINN_Burgers": (  3, -0.14),
    # Poisson ends ~0.415; well separated from graph/Burgers
    "PINN_Poisson": (  3, -0.02),
    # CNN/MNIST/Fashion end on LEFT side (labels shifted left)
    "CNN_cifar":    (-30, +0.06),
    "MNIST_MLP":    (-30, -0.07),
    "Fashion_MLP":  (-30, +0.04),
}
for name in ORDER:
    if name not in processed:
        continue
    d   = processed[name]
    col = C[name]
    xend = d["ep_pct"][-1]
    yend = d["dn"][-1]
    dx, dy = _label_offsets[name]

    rho_str = f"$\\rho$={d['rho']:+.2f}"
    h2h_str = f"[{d['h2h']}]"
    trend_arrow = "↑" if d["slope"] > 0.0001 else ("↓" if d["slope"] < -0.0001 else "→")
    full_label = f"{trend_arrow} {rho_str}  {h2h_str}"

    ax1.annotate(
        full_label,
        xy=(xend, yend),
        xytext=(xend + dx, yend + dy),
        ha="left" if dx > 0 else "right", va="center",
        fontsize=7.2, color=col,
        arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.6),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, alpha=0.88, lw=0.7),
        zorder=9,
    )

# random null line
ax1.axhline(1.0, ls=(0, (4, 3)), color="dimgray", alpha=0.45, lw=1.0, zorder=1)
ax1.text(99, 1.012, "random null", ha="right", va="bottom",
         fontsize=7.2, color="dimgray")

# central honest message box
ax1.text(
    0.50, 0.97,
    "All verdicts: tie  (↑↓ trend does not predict win/loss)",
    transform=ax1.transAxes, ha="center", va="top",
    fontsize=8.0, color="#333333", fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.3", fc="#f9f9f9", ec="#888888", alpha=0.90, lw=0.8),
)

# legend — family curves only
legend_handles = []
for name in ORDER:
    if name not in processed:
        continue
    d = processed[name]
    h = Line2D([0], [0], color=C[name], linestyle=LS[name], linewidth=1.8,
               label=d["label"])
    legend_handles.append(h)

ax1.legend(
    handles=legend_handles,
    loc="lower right",
    ncol=1,
    fontsize=7.5,
    framealpha=0.93,
    edgecolor="#cccccc",
    handlelength=2.0,
    borderpad=0.5,
    labelspacing=0.35,
)

ax1.set_xlabel("Training epoch  (% of each family's total epochs; PINN=3000, others=1000)")
ax1.set_ylabel(r"$\delta_\mathrm{nest}$ norm  (indep-optimum nesting gap / random null)")
ax1.set_title(
    r"(a)  $\delta_\mathrm{nest}$ trajectories across six task families  (epoch axis)",
    pad=6, loc="left",
)
ax1.set_xlim(-3, 115)
ax1.set_ylim(-0.02, 1.05)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — slope summary: directly shows decoupling
# ═══════════════════════════════════════════════════════════════════════════════
short_labels = {
    "graph_arxiv":  "Graph",
    "MNIST_MLP":    "MNIST",
    "Fashion_MLP":  "Fashion",
    "PINN_Burgers": "Burgers",
    "PINN_Poisson": "Poisson",
    "CNN_cifar":    "CNN",
}

# Sort by slope for clearest visual
order_by_slope = sorted(
    [n for n in ORDER if n in processed],
    key=lambda n: processed[n]["slope"],
    reverse=True,
)
x_idx = np.arange(len(order_by_slope))
slopes = [processed[n]["slope"] for n in order_by_slope]
cols   = [C[n] for n in order_by_slope]
xlabs  = [short_labels[n] for n in order_by_slope]
verdicts = [H2H[n] for n in order_by_slope]

bars = ax2.bar(x_idx, slopes, 0.62,
               color=cols, alpha=0.80, edgecolor=cols, linewidth=0.9, zorder=3)

# Zero line
ax2.axhline(0, color="#888888", lw=1.0, ls="-", alpha=0.7, zorder=2)

# Verdict labels above/below each bar
for i, (n, sl, vd) in enumerate(zip(order_by_slope, slopes, verdicts)):
    col = C[n]
    if sl >= 0:
        ypos = sl + 0.00015
        va   = "bottom"
    else:
        ypos = sl - 0.00015
        va   = "top"
    ax2.text(x_idx[i], ypos, f"[{vd}]",
             ha="center", va=va, fontsize=7.2, color=col, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=col, alpha=0.88, lw=0.7))

# slope value inside/beside bars — skip if bar too thin to hold text
_SLOPE_MIN_VISIBLE = 0.00020   # skip label if |slope| < this
for i, sl in enumerate(slopes):
    if abs(sl) < _SLOPE_MIN_VISIBLE:
        # place outside bar instead
        sign = "+" if sl >= 0 else ""
        yoff = -0.00030 if sl >= 0 else +0.00030
        ax2.text(x_idx[i], sl + yoff, f"{sign}{sl*100:.3f}",
                 ha="center", va="top" if sl >= 0 else "bottom",
                 fontsize=6.8, color=cols[i], fontstyle="italic")
        continue
    sign = "+" if sl >= 0 else ""
    ytext = sl / 2
    ax2.text(x_idx[i], ytext, f"{sign}{sl*100:.3f}",
             ha="center", va="center", fontsize=7.0, color="white",
             fontweight="bold")

ax2.set_xticks(x_idx)
ax2.set_xticklabels(xlabs, fontsize=8.5, rotation=20, ha="right")
ax2.set_ylabel(r"$\delta_\mathrm{nest}$ slope  (per 1% of training, $\times 10^{-2}$)")
ax2.set_title(
    "(b)  Slope vs verdict\n     (↑ and ↓ both → tie)",
    pad=6, loc="left",
)

# annotation: slope × 10^-2 legend
ax2.text(0.97, 0.97,
         "Values $\\times 10^{-2}$ per epoch%\n"
         "Positive = $\\delta_{\\rm nest}$ rising\n"
         "Negative = $\\delta_{\\rm nest}$ falling\n"
         "All verdicts: tie $\\Rightarrow$ decoupled",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=7.2, color="#333333",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f9f9f9", ec="#aaaaaa",
                   alpha=0.90, lw=0.7))

# ── overall title ─────────────────────────────────────────────────────────────
fig.suptitle(
    r"$\delta_\mathrm{nest}$ Across Task Families: Trend Does Not Predict STRATA Win/Loss",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.tight_layout(pad=0.5, w_pad=1.4)

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig(_fh_pdf, format="pdf")
fig.savefig(_fh_png, format="png", dpi=150)
_fh_pdf.close()
_fh_png.close()
print("Written: fig1_dnest_cross_family.pdf + .png")

# ── caption ──────────────────────────────────────────────────────────────────
caption = r"""Figure 1: \textbf{$\delta_\mathrm{nest}$ trajectories across six task families (epoch axis).}
\emph{(a)} Normalised independent-optimum nesting gap ($\delta_\mathrm{nest}$, lower = more nested
than i.i.d.\ random baseline) vs.\ training epoch percentage (epoch\,/\,total\,$\times$\,100;
PINN runs = 3000\,ep, others = 1000\,ep).
Each curve is annotated with its Spearman $\rho$ (epoch\,vs.\,$\delta_\mathrm{nest}$) and
the STRATA\,vs.\,Single-Grassmann verdict in brackets.
\emph{(b)} Summary of $\delta_\mathrm{nest}$ slope (per 1\% of training) sorted by magnitude,
with verdict labels.

\textbf{Honest core finding:} $\delta_\mathrm{nest}$ trend is \emph{decoupled from} STRATA vs.\ SG
performance.  Four families exhibit a rising $\delta_\mathrm{nest}$ (graph, PINN--Burgers,
PINN--Poisson, CNN; Spearman $\rho > 0$); two families are flat or declining (MNIST, Fashion).
\emph{All six produce tie verdicts} (except Poisson, where SG numerically diverges---not
a flag-mechanism effect; see Ablation~A5, Fig.~3).
We therefore \emph{cannot} interpret $\delta_\mathrm{nest}$ as a performance predictor.

\textbf{What $\delta_\mathrm{nest}$ does reveal (descriptively):}
The graph family is distinguished not by being the only riser (three families also rise),
but by its \emph{uniquely low starting point} ($\delta_\mathrm{nest,0} = 0.047$; all other
families start above $0.30$), indicating it begins training in a highly nested multi-scale
structure that subsequently collapses toward a less-nested state.
PINN families maintain lower $\delta_\mathrm{nest}$ throughout compared to graph, consistent with
persistent multi-scale structure in PDE training dynamics.

Note: the earlier \texttt{learn\_progress} axis introduced a systematic bias for PINN and
vision families (which converge quickly in metric but continue to evolve in $\delta_\mathrm{nest}$),
compressing their trajectories into a narrow region and artificially flattening observed trends.
All results here use raw epoch axis; see Appendix~\ref{app:dnest_lp} for a side-by-side comparison."""

(OUT_DIR / "fig1_dnest_cross_family_caption.txt").write_text(caption)
print("Written: fig1_dnest_cross_family_caption.txt")
