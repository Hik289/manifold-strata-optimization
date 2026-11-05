from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=root/'analysis/revision_20261101'
pinn=json.loads((out/'pinn_selection_audit.json').read_text()); cl=json.loads((out/'classification_selection_audit.json').read_text())
rows=[]
for task in ['burgers','burgersg2','poisson']:
 a={x['method']:min([v for v in x['cells'] if v[2].endswith('R64')],key=lambda v:v[0]) for x in pinn if x['task']==task}
 rows.append((task,a['single_grassmann'][1],a['strata'][1],False))
for task in ['mnist','fashion','cifar10','cifar100']:
 a={m:max([x for x in cl if x['task']==task and x['method']==m],key=lambda x:x['val_acc']) for m in ['single_grassmann','strata']}
 rows.append((task,a['single_grassmann']['test_acc'],a['strata']['test_acc'],True))
fig,axes=plt.subplots(1,2,figsize=(9,3.1),layout='constrained')
for ax,subset,title in [(axes[0],rows[:3],'PDE-selected PINNs, rank 64'),(axes[1],rows[3:],'Validation-selected classifiers')]:
 x=range(len(subset));ax.bar([v-.18 for v in x],[v[1] for v in subset],.36,label='Single-Grassmann',color='#7892ad');ax.bar([v+.18 for v in x],[v[2] for v in subset],.36,label='STRATA',color='#c38156');ax.set_xticks(list(x),[v[0] for v in subset],rotation=15);ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
 if subset[0][3]:ax.set_ylim(.65,1);ax.set_ylabel('Test accuracy (higher is better)')
 else:ax.set_yscale('log');ax.set_ylabel('Relative L2 (lower is better)')
axes[0].legend(frameon=False,fontsize=8);fig.savefig(root/'paper/figures/fig4_validation_audited.pdf');plt.close(fig)
tex=r'''\section{Additional Numerical Checks and Protocol Audit}
\label{app:revision-audit}
\paragraph{Synthetic checks of the corrected theory.}
All synthetic random starts use a fixed generator initialized with seed 42; multiple starts here are landscape diagnostics, not additional benchmark seeds. For the explicit commuting example in Proposition~\ref{prop:tf-cycle}, 100 random orthogonal starts followed by 2000 projected-ascent steps at step size 0.05 give 88 global solutions (objective 6) and 12 strict nonglobal local maxima (objective 5). Finite-difference second derivatives at the latter are approximately $(-1,-2,-1)$ in the three plane-rotation coordinates. This validates the counterexample, not a universal basin probability. For the basis-dependence example, with $\epsilon=10^{-6}$ the identity-basis update is approximately $(0.999999,0)$ and the rotated-basis update is $(1.414212,0)$, a difference of 0.414213. A deterministic diagonal quadratic check with Hessian $\operatorname{diag}(1,2,4)$ and preconditioner $\operatorname{diag}(0.5,1,1.5)$ satisfies the descent inequality at every one of 1000 steps; squared gradient decreases from 21 to $8.98\times10^{-7}$. These checks exercise algebra and local behavior, not the stochastic assumptions of real neural training.

\paragraph{Recomputed selection audit.}
We independently recomputed terminal-100-epoch means from raw logs. Classification configurations are selected only by validation accuracy; the resulting Fashion-MNIST Single-Grassmann test accuracy is 0.9033, correcting the previous 0.9025. For PINNs, pure terminal PDE-loss selection chooses Poisson STRATA lr $10^{-3}$, rank 64, with relative $L_2$ error 0.1874. The previously highlighted 0.0603 belongs to lr $5\times10^{-4}$, rank 32 and is not the minimum-PDE-loss configuration. It remains a diagnostic configuration, not our primary validation-selected endpoint. The selected Single-Grassmann Poisson configuration gives 2.0028; the grid also contains nonfinite outcomes. The new result supports a limited robustness observation rather than a general optimization advantage. Historical figures and full grids are retained; the audited figure and main table use the corrected selection.

\begin{figure}[ht]
\centering
\includegraphics[width=.98\linewidth]{figures/fig4_validation_audited.pdf}
\caption{Recomputed endpoints with validation-based configuration selection. Log scale is used for PINN errors. No confidence intervals are claimed for the fixed-seed cases.}
\end{figure}

\paragraph{Proposal coverage.}
The original six broad proposal families must not be confused with the eight existing dataset rows. Existing completed results cover portions of vision, graph learning and scientific ML; the original archive has no completed language-pretraining, continual-learning or artificial-distribution-shift family. The November revision launches an isolated training host queue for full-data Rotated-MNIST continual learning and shifts, Split-CIFAR-100 replay, CIFAR distribution shifts and WikiText-103 GPT-2 experiments. New queue entries, smoke tests and partial trajectories are not counted as completed results. Generated Darcy/FNO and additional citation-graph adapters are separate from existing archived evidence. Completion of the full proposal additionally requires the remaining image, continual-learning, corpus and ablation matrix; no missing cell is replaced with an inferred result.
'''
if not (root/'paper/sections/revision_audit.tex').exists():
 (root/'paper/sections/revision_audit.tex').write_text(tex)
