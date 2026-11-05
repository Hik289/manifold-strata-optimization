"""A comparison row requires every method at the stated complete search budget."""
import collections,json,math
from pathlib import Path
import numpy as np
from paper_table_style import format_tex
root=Path(__file__).resolve().parents[2];folder=root/'analysis/revision_20261101/breadth';live=folder/'live';q=json.loads((live/'queue_snapshot.json').read_text());groups=collections.defaultdict(dict);methods=['strata','single','adamw','galore','independent']
for j in q:
 if j['status']!='complete':continue
 d=live/j['out'];cfg=json.loads((d/'config.json').read_text());rows=[json.loads(s) for s in (d/'log.jsonl').read_text().splitlines()];tail=rows[-100:]
 if len(rows)<1000 or 'val_acc' not in tail[-1] or cfg.get('ablation','full')!='full':continue
 ds=cfg.get('dataset',cfg.get('task'));model=cfg.get('model',cfg.get('architecture','MLP'));key=(ds,model,cfg.get('base',''));method=cfg['method']
 if method not in methods:continue
 lr=float(cfg['lr']);groups[key][(method,lr)]=dict(id=j['id'],method=method,lr=lr,val=float(np.mean([r['val_acc'] for r in tail])),test=float(np.mean([r['test_acc'] for r in tail])),source=j['out'])
cohorts=[]
for key,values in sorted(groups.items()):
 rates=[.0003,.001]
 if all((m,lr) in values for m in methods for lr in rates):
  chosen={m:max([values[(m,lr)] for lr in rates],key=lambda r:r['val']) for m in methods};protocol='2-rate val';trials=10
 elif all((m,.0003) in values for m in methods):chosen={m:values[(m,.0003)] for m in methods};protocol='fixed 3e-4';trials=5
 else:continue
 cohorts.append(dict(dataset=key[0],model=key[1],base=key[2],protocol=protocol,trials=trials,methods=chosen,strata_minus_sg=chosen['strata']['test']-chosen['single']['test']))
(folder/'complete_cohorts.json').write_text(json.dumps(cohorts,indent=2))
escape=lambda s:str(s).replace('_','\\_')
tex=r'''\paragraph{New complete comparison cohorts.}
Table~\ref{tab:new-cohorts} includes a cohort only when all five methods have completed the stated budget. Two-rate rows require all ten trials before validation-based learning-rate selection; fixed-rate rows use $3\times10^{-4}$ for every method. An incomplete search is not reported as a validation-selected winner.
\begin{table}[t]
\centering\small
\caption{New full-budget comparison cohorts, seed 42, terminal-100 test accuracy. A two-rate search selects each method using terminal validation accuracy. Fixed-rate rows use no rate selection. Only complete five-method cohorts are shown; the full ledger retains every completed trial.}
\label{tab:new-cohorts}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrr}\toprule
Task/model & Protocol & STRATA & Single-G & AdamW & GaLore & Indep.\\\midrule
'''
for c in cohorts:
 label=f"{escape(c['dataset'])}/{escape(c['model'])}"
 if c['dataset']=='rotated_mnist_cl':label='Rotated-MNIST/MLP ('+{'finetune':'fine-tuning','replay':'replay'}.get(c['base'],escape(c['base']))+')'
 best=max(c['methods'][m]['test'] for m in methods)
 cells=[(r'\textbf{'+f"{c['methods'][m]['test']:.4f}"+'}') if c['methods'][m]['test']==best else f"{c['methods'][m]['test']:.4f}" for m in methods]
 tex+=label+' & '+c['protocol']+' & '+' & '.join(cells)+r' \\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
for c in cohorts:
 if c['dataset']=='Cora' and c['model']=='gcn' and c['protocol']=='2-rate val':
  tex+=r'After completing both rates on Cora/GCN, validation selects STRATA $3\times10^{-4}$ and Single-Grassmann $10^{-3}$. Their test accuracies are 0.70120 and 0.71643; the earlier positive fixed-rate difference therefore does not survive the completed search. GaLore reaches 0.77180 in this cohort. This adverse comparison is retained.'+'\n'
 if c['dataset']=='PubMed' and c['model']=='sage' and c['protocol']=='2-rate val':
  chosen=c['methods'];s=chosen['strata']['test'];g=chosen['single']['test'];best=max(chosen,key=lambda m:chosen[m]['test'])
  tex+=f'On the completed PubMed/SAGE two-rate cohort, STRATA reaches {s:.5f} versus Single-Grassmann {g:.5f}, a descriptive gain of {100*(s-g):.3f} percentage points. The highest test accuracy among the validation-selected methods is {chosen[best]["test"]:.5f} ({best}); this single-seed observation does not establish a graph-family advantage.'+'\n'
 if c['dataset']=='rotated_mnist_cl' and c['base']=='replay' and c['protocol']=='2-rate val':
  tex+=r'\paragraph{Completed Rotated-MNIST replay protocol.} Both fine-tuning and replay use the same seed-42 split (55000 training, 5000 validation and 10000 test examples), two 256-wide hidden layers, rank 64, and ten rotations from $0^\circ$ to $90^\circ$ in $10^\circ$ increments. Each rotation receives 100 epochs. Replay stores 200 examples after each task and adds one recalled example per four current-task examples in each minibatch (current batch size 512); all five methods use this same replay rule. Endpoints average epochs 901--1000, evaluating all ten seen rotations. Every method completes both original rates before terminal validation accuracy selects the rate: $10^{-3}$ for STRATA, AdamW, GaLore and Independent, and $3\times10^{-4}$ for Single-Grassmann. STRATA improves on Single-Grassmann by 1.412 percentage points and on Independent by 1.443 points; AdamW and GaLore remain higher. These single-seed comparisons do not establish statistical significance. The loss curves retain all selected methods without smoothing; the mean validation loss covers an increasing set of seen rotations at task boundaries.'+'\n'
(root/'paper/sections/breadth_results.tex').write_text(format_tex(tex))
# Plain convergence summaries are descriptive; logged norms are minibatch diagnostics.
audit=json.loads((folder/'loss_curve_audit.json').read_text());lines=[r'''\begin{longtable}{llrrrr}
\caption{Selected classifier convergence diagnostics. AUC integrates raw training loss over the observed archived epoch indices (0--999); $g_{\rm log}$ is the mean logged gradient norm, not a claim to the exact full-objective gradient. Missing gradient norms were not recorded in those archives and are left absent. Configurations use the selection rule of Figure~\ref{fig:loss-classification}.}\\
\toprule Dataset & Method & Loss AUC & Train loss & Val loss & $g_{\rm log}$\\\midrule\endfirsthead
\toprule Dataset & Method & Loss AUC & Train loss & Val loss & $g_{\rm log}$\\\midrule\endhead
''']
for r in audit['summaries']:
 rows=[json.loads(s) for s in (root/r['source']).read_text().splitlines()];vl=float(np.mean([z['val_loss'] for z in rows[-100:]]));vals=[r['loss_auc'],r['terminal100_train_loss'],vl,r['terminal100_gradient_norm']];lines.append(escape(r['dataset'])+' & '+escape(r['method'])+' & '+' & '.join('--' if v is None else f'{v:.3g}' for v in vals)+r' \\'+'\n')
lines.append(r'\bottomrule\end{longtable}'+'\n');(root/'paper/sections/convergence_metrics_table.tex').write_text(format_tex(''.join(lines)));print(json.dumps(cohorts,indent=2))
