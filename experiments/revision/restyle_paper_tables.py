"""Rebuild non-loss visualizations as auditable tables and redraw retained losses."""
import argparse,collections,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_table_style import format_tex,loss_style

p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));p.add_argument('--snapshot',default='snapshot_light');a=p.parse_args()
root=Path(a.root).resolve();snapshot=root/a.snapshot;sec=root/'paper/sections';report=root/'analysis/revision_20261102/icml_tables';report.mkdir(parents=True,exist_ok=True)
loss_style(plt);curves=[];conversions=[]
colors=dict(strata='#D55E00',single='#0072B2',adamw='#333333',galore='#009E73',independent='#CC79A7')
names=dict(strata='STRATA',single='Single-G',adamw='AdamW',galore='GaLore',independent='Independent')
def table(caption,label,columns,header,rows,wide=False):
    env='table*' if wide else 'table'
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    return ('\\begin{'+env+'}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+' & '.join(header)+r' \\'+'\n\\midrule\n'+body+'\n\\bottomrule\n\\end{tabular}\n\\end{'+env+'}\n')
def replace_figure(name,replacement):
    hits=0
    for path in sec.glob('*.tex'):
        text=path.read_text()
        def sub(m):
            nonlocal hits
            if name in m.group(0):hits+=1;return replacement
            return m.group(0)
        changed=re.sub(r'\\begin\{figure\*?\}(?:\[[^\]]*\])?[\s\S]*?\\end\{figure\*?\}',sub,text)
        if changed!=text:path.write_text(changed)
    labels=re.findall(r'\\label\{([^}]+)\}',replacement)
    conversions.append(dict(asset=name,replaced_this_pass=hits,present_as_tables=all(any(r'\label{'+label+'}' in p.read_text() for p in sec.glob('*.tex')) for label in labels)))
def logpath(source):
    old=Path(source)
    candidates=[old,root/old,snapshot/'runs_revision'/old.name/'log.jsonl',snapshot/'runs_revision'/old.parent.name/'log.jsonl']
    return next(q for q in candidates if q.is_file())
def draw(name,items,keys,labels):
    fig,axs=plt.subplots(1,len(keys),figsize=(7,2.45),squeeze=False,layout='constrained')
    for method,source in items:
        path=logpath(source);rows=[json.loads(z) for z in path.read_text().splitlines()];x=[r.get('epoch',r.get('unit')) for r in rows]
        for ax,key,label in zip(axs.flat,keys,labels):
            values=[r[key] if isinstance(key,str) else r[key[0]][key[1]] for r in rows]
            ax.plot(x,values,color=colors[method],label=names[method],lw=.95)
            ax.set(xlabel='Epoch',ylabel=label,yscale='log');ax.grid(alpha=.18)
            curves.append(dict(figure=name,source=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metric=key,points=len(rows),smoothing='none'))
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.1))
    for ext in ['pdf','png']:fig.savefig(root/'paper/figures'/(name+'.'+ext),dpi=190)
    plt.close(fig)

# Archive the expanded main-text theory bridge; retain every existing theorem/proof.
bridge=sec/'theory_main_expanded.tex'
if not bridge.exists():
    bridge.write_text((sec/'theory_overview.tex').read_text().replace(r'\section{Convergence and Generalization}',r'\section{Expanded Convergence and Generalization Bridge}').replace(r'\label{sec:theory}',r'\label{app:expanded-theory-bridge}'))
(sec/'theory_overview.tex').write_text(r'''\section{Convergence and Generalization}\label{sec:theory}
We analyze the chain from covariance estimation to learned flags, evolving adaptive states, parameter descent, and generalization. Complete hypotheses and proofs are in Appendix~\ref{app:theory-full}; Appendix~\ref{app:expanded-theory-bridge} retains the expanded mathematical bridge.

\paragraph{Flag geometry and tracking.}
For $P_k=U_{:r_k}U_{:r_k}^\top$, put $\Pi_k=P_k-P_{k-1}$ and $\Phi(U)=\sum_k\omega_k\operatorname{tr}(P_kC_k)$. The overlap coupling $T_{ja}=\operatorname{tr}(\Pi_j(U)\Pi_a(V))$ gives the exact distance identity
\begin{equation}
 d_P^2(U,V)=\sum_{j,a}|j-a|T_{ja}.
\end{equation}
Charging each misplaced direction only for the prefix boundaries it crosses gives the sharp factor $K$ in the assignment-distance comparison. Hoffman--Kruskal integrality characterizes the commuting-covariance global optimum as a capacity-constrained assignment (Theorem~\ref{thm:tf-assignment}). A commuting counterexample still has a strict nonglobal local maximum (Proposition~\ref{prop:tf-cycle}). Under the regular local-chart hypotheses, the oracle perturbation and tracking estimates are
\begin{align}
 \|x_{\widehat C}^\star-x_C^\star\|
 &\le\frac{2J_\psi}{\mu_\phi}\sum_k\omega_k\sqrt{r_k}\epsilon_k,\\
 \sqrt{E_s}&\le q^s\sqrt{E_0}+\sum_{j<s}q^{s-1-j}b_j.
\end{align}
Here $\epsilon_k$ bounds covariance error and $b_j$ combines statistical, sketch, and drift forcing. The sharp quadratic contraction is $q_\star=(L_\phi-\mu_\phi)/(L_\phi+\mu_\phi)$. Centered noise and persistent drift have different gains, $\xi^2/(1-q^2)$ and $1/(1-q)$, respectively (Theorem~\ref{thm:tf-tracking}).

\paragraph{Convergence with evolving adaptive state.}
For a predictable preconditioner $mI\preceq B_t\preceq MI$, write $w_{t+1}=w_t-\eta_t(B_tg_t+e_t)$ and retain the exact discrepancy $e_t$ from the practical update. With $S_T=\sum_{t<T}\eta_t$, the nonconvex estimate is
\begin{align}
 \frac{\sum_{t<T}\eta_t\mathbb E\|\nabla F(w_t)\|^2}{S_T}
 &\le \frac{2\Delta_0}{mS_T}
 +\frac{2L\sum_{t<T}\eta_t^2\mathbb E\mathsf v_t}{mS_T}\notag\\
 &\quad+\mathcal E_T,
\end{align}
where $\mathcal E_T$ is the explicit bias, transport, and normalization discrepancy term in Theorem~\ref{thm:tf-convergence}; $\mathsf v_t=\operatorname{tr}(B_t^2\Sigma_t)$ retains directional noise rather than replacing it by $M^2\sigma_t^2$. Under the PL condition, the initial gap contracts by $(1-\mu m\eta)^T$ toward a stated noise/discrepancy floor (Theorem~\ref{thm:tf-PL}). The implemented current-gradient normalizer requires separate analysis: Theorem~\ref{thm:actual-coercivity} controls its conditional symmetry defect, and Theorem~\ref{thm:actual-convergence} propagates observable moment/reset/transport budgets. The explicit asymmetric-noise stationary bias in Proposition~\ref{prop:actual-skew-counterexample} explains why gradient unbiasedness alone is insufficient.

\paragraph{Generalization and directional complexity.}
Coupling neighboring samples gives a five-component recursion for parameters, flags, sketches, first moments, and square-root second moments. Its nonnegative transition matrices $\mathcal M_t$ yield
\begin{equation}
 \epsilon_T^{\rm stab}\le\frac{L_0}{n}e_w\sum_{t<T}b_t
 (\mathcal M_{T-1}\cdots\mathcal M_{t+1})d_t.
\end{equation}
This accounts for the whole optimizer state (Theorem~\ref{thm:rigid-generalization}). Independently, the Seeger--Maurer route keeps inverse binary KL and the exact moment constant $\Xi_n$, avoiding a Pinsker relaxation (Theorem~\ref{thm:rigid-pac}). Under the stated bounded-loss curvature hypothesis, optimizing a full-rank Gaussian posterior aligned with the flag gives
\begin{align}
 \mathcal C_\lambda(U)
 &=\frac1{2\lambda}\sum_jmq_j\log(1+2\lambda\sigma_0^2\overline h_j(U)),\\
 (\sigma_j^2)^\star&=\frac{\sigma_0^2}{1+2\lambda\sigma_0^2\overline h_j(U)}.
\end{align}
Refining a block with unequal average curvatures strictly decreases this optimized complexity; upper and lower gains depend on their weighted variance (Proposition~\ref{prop:directional-refinement}). Tracking sensitivity depends on spectral spread, with isotropic curvature canceling exactly. These are conditional guarantees with explicit hypotheses, rather than empirical certificates of those hypotheses on every benchmark.
''')
ap=sec/'appendix.tex';t=ap.read_text()
if r'\input{sections/theory_main_expanded}' not in t:t=t.replace(r'\input{sections/theory_full}',r'\input{sections/theory_full}'+'\n'+r'\input{sections/theory_main_expanded}')
ap.write_text(t)

# The old bar-chart endpoints come from the same retained validation audit.
base=root/'analysis/revision_20261101'
pinn=json.loads((base/'pinn_selection_audit.json').read_text());cl=json.loads((base/'classification_selection_audit.json').read_text());rows=[]
for task in ['burgers','burgersg2','poisson']:
    chosen={x['method']:min([v for v in x['cells'] if v[2].endswith('R64')],key=lambda v:v[0]) for x in pinn if x['task']==task}
    vals=[chosen[m][1] for m in ['single_grassmann','strata']];best=min(vals)
    rows.append([task,r'Rel-$L_2$ $\downarrow$',*[r'\textbf{'+f'{v:.5f}'+'}' if v==best else f'{v:.5f}' for v in vals]])
for task in ['mnist','fashion','cifar10','cifar100']:
    chosen={m:max([x for x in cl if x['task']==task and x['method']==m],key=lambda x:x['val_acc']) for m in ['single_grassmann','strata']}
    vals=[chosen[m]['test_acc'] for m in ['single_grassmann','strata']];best=max(vals)
    rows.append([task,r'Accuracy $\uparrow$',*[r'\textbf{'+f'{v:.5f}'+'}' if v==best else f'{v:.5f}' for v in vals]])
replace_figure('fig4_validation_audited',table(r'\textbf{Validation-audited head-to-head endpoints.} PINNs restrict both methods to rank 64 and select by terminal PDE loss. Classifiers select by validation accuracy. Bold compares the two methods; all values use seed 42.','tab:head-audited','llrr',['Task','Metric','Single-G','STRATA'],rows))

data=json.loads((root/'experiments/dnest_cross_family.json').read_text());rows=[]
for key,label in [('graph_arxiv','ogbn-arxiv'),('MNIST_MLP','MNIST'),('Fashion_MLP','Fashion'),('PINN_Burgers','Burgers'),('PINN_Poisson','Poisson'),('CNN_cifar','CIFAR')]:
    tr=data[key]['dnest_trajectory'];mx=max(r['epoch'] for r in tr);ys=[r['dnest_norm'] for r in tr];early=np.mean([r['dnest_norm'] for r in tr if r['epoch']<mx*.5]);late=np.mean([r['dnest_norm'] for r in tr if r['epoch']>=mx*.5]);rho=spearmanr([r['epoch'] for r in tr],ys).statistic
    rows.append([label,f'{ys[0]:.3f}',f'{ys[-1]:.3f}',f'{early:.3f}',f'{late:.3f}',f'{rho:+.3f}'])
replace_figure('fig1_dnest_cross_family',table(r'\textbf{Epoch-axis nesting diagnostics.} First/last values and early/late-half means use the complete retained diagnostic trajectory. Spearman $\rho$ uses recorded epochs. Diagnostic trends do not establish a performance ranking.','tab:dnest','lrrrrr',['Task','First','Last','Early mean','Late mean',r'$\rho$'],rows))

bug=table(r'\textbf{Numerical specification fixes.} The MNIST epsilon check is a diagnostic configuration; gradient norm and accuracy are the recorded values for that check.','tab:bugs','lrrr',[r'Projected $\epsilon$',r'Max. $\|g\|$',r'Test acc. $\uparrow$','Separate counters'],[[r'$10^{-8}$',r'$1.87\times10^5$','0.686','--'],[r'$10^{-6}$','14.2',r'\textbf{0.969}','--']],wide=False)
bug+=table(r'\textbf{Counter handling at a flag refresh.} Transported mature first moments and reinitialized second moments have different statistical ages.','tab:counter-fix','lll',['Rule','First-moment age','Second-moment age'],[['Shared reset','Reset despite transport','Reset'],['Separate counters','Retain mature age','Reset']],wide=False)
replace_figure('fig3_spec_bugs_before_after',bug)

profile=json.loads((root/'analysis/revision_20261102/experimental_evidence/xi_profile.json').read_text())
rows=[[names[r['method']],f"{r['xi']:g}",f"{r['lr']:.0e}",f"{r['val']:.5f}",f"{r['test']:.5f}"] for r in profile]
replace_figure('fig_evidence_xi.pdf',table(r'\textbf{Complete flag-step sensitivity profiles.} Each point selects its learning rate on validation after both candidates finish. Test values are reporting only.','tab:xi-profile','lrrrr',['Method',r'$\xi$','lr','Val acc','Test acc'],rows))

summary=json.loads((base/'breadth/actual_theory_validation.json').read_text());trace=json.loads((base/'breadth/actual_state_budget_trace.json').read_text());sigma=.3;rows=[]
for eps in [1e-4,1e-3,.01,.1,1.,10.]:
    v=8*sigma/3+5*eps/4;stationary=2*sigma*sigma/(math.sqrt(v*v+4*sigma*sigma)+v)
    rows.append([f'{eps:g}','0',f'{stationary:.8f}'])
normalization=table(r'\textbf{Exact normalization-bias counterexample.} Symmetric and asymmetric zero-mean noises have variance $\sigma^2$, $\sigma=.3$. The true optimum is zero. The asymmetric stationary root is given by Proposition~\ref{prop:actual-skew-counterexample}.','tab:actual-normalization','rrr',[r'$\epsilon$','Symmetric root','Asymmetric root'],rows)
normalization+=table(r'\textbf{Finite-trajectory diagnostics at $\epsilon=.01$.} Means use the final 5000 of 30000 seed-42 stochastic-approximation iterates. These are finite-run observations, not asymptotic estimates.','tab:normalization-tail','lr',['Update / noise','Final-5000 mean $w$'],[[k,f'{v:.8f}'] for k,v in summary['tail_mean'].items()])
replace_figure('fig_actual_normalization_theory',normalization)
rows=[]
for label,observed,bound in [('Preconditioner','B_change','B_bound'),('Momentum coefficient','theta_change','theta_bound'),('Transport defect','u','u_bound')]:
    ts=[r for r in trace if observed in r];slacks=[r[bound]-r[observed] for r in ts]
    rows.append([label,len(ts),f'{max(r[observed] for r in ts):.3e}',f'{max(r[bound] for r in ts):.3e}',f'{min(slacks):+.3e}',sum(s<-1e-10 for s in slacks)])
budgets=table(r'\textbf{Implementation-level variation-budget checks.} The float64 trajectory has 100 steps and 20 flag refreshes. Maxima summarize each sequence separately; inequalities are checked step by step. A violation means slack below $-10^{-10}$.','tab:actual-state-budgets','lrrrrr',['Quantity','Steps','Max observed','Max bound','Min slack','Violations'],rows)
budgets+=table(r'\textbf{Exact adaptive-update identities.} Maximum residual over the same 100-step trajectory.','tab:actual-identities','lr',['Identity','Maximum residual'],[[k.replace('_',' '),f'{v:.3e}'] for k,v in summary['actual_implementation_identity_errors'].items()])
replace_figure('fig_actual_state_budgets',budgets)

# Redraw mixed diagnostic figures using genuine loss channels only.
ledger=list(csv.DictReader((root/'results_revision/completed_matrix.csv').open()))
for task,model,name,keys,labels in [
    ('PubMed','sage','fig_evidence_pubmed',['train_loss','val_loss'],['Training cross-entropy','Validation cross-entropy']),
    ('Darcy-generated','FNO','fig_evidence_darcy',['train_loss',('val','loss'),('test','loss')],['Training MSE','Validation MSE','Test MSE'])]:
    items=[]
    for method in colors:
        candidates=[r for r in ledger if r['task']==task and r['model']==model and r['method']==method and r['ablation']=='full']
        best=(max if candidates[0]['metric']=='acc' else min)(candidates,key=lambda r:float(r['val']));items.append((method,best['source']))
    draw(name,items,keys,labels)

refs={'Figure~\\ref{fig:head}':'Table~\\ref{tab:head-audited}','Figure~\\ref{fig:dnest}':'Table~\\ref{tab:dnest}','Figure~\\ref{fig:bugs}':'Tables~\\ref{tab:bugs} and~\\ref{tab:counter-fix}'}
main_tables={'related_work_table','positive_case_studies','evidence_main_live'}
for path in sec.glob('*.tex'):
    text=path.read_text()
    for old,new in refs.items():text=text.replace(old,new)
    text=text.replace('Historical figures and full grids are retained; the audited figure and main table use the corrected selection.','The archived full grids and audited endpoint tables use the corrected selection.')
    text=text.replace('The corrected endpoints are visualized in','The corrected endpoints are reported in')
    text=text.replace('Left: physics training loss; right: relative $L_2$ error.','Left: total training loss; right: PDE residual loss. Exact-solution endpoint errors are tabulated separately.')
    text=text.replace('Complete 2000-epoch Darcy/FNO trajectories.','Complete 2000-epoch Darcy/FNO loss trajectories: training, validation, and test MSE.')
    if path.stem in main_tables:
        # Wide tables use the ICML double-column float; the loss float remains single-column.
        text=text.replace(r'\begin{table}[',r'\begin{table*}[').replace(r'\end{table}',r'\end{table*}')
    if path.stem=='evidence_main_live':
        text=text.replace(r'\begin{figure}[',r'\begin{figure*}[').replace(r'\end{figure}',r'\end{figure*}')
    text=format_tex(text)
    path.write_text(text)
assert not any('limitation' in p.read_text().lower() for p in sec.glob('*.tex'))
(report/'visual_conversion_audit.json').write_text(json.dumps(dict(conversions=conversions,redrawn_losses=curves,policy='Figures contain loss trajectories only; all other displays are tables. Original assets and raw logs retained.'),indent=2))
print(json.dumps(dict(converted_displays=sum(x['present_as_tables'] for x in conversions),redrawn_loss_curves=len(curves))))
