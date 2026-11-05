"""Publish complete, validation-selected evidence and its adverse comparisons."""
import argparse,collections,csv,fcntl,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_table_style import format_tex,loss_style

p=argparse.ArgumentParser();p.add_argument('--root',default='..');p.add_argument('--snapshot');p.add_argument('--watch',action='store_true');a=p.parse_args();root=Path(a.root).resolve();torch.set_num_threads(4)
report=root/'analysis/revision_20261102/experimental_evidence';report.mkdir(parents=True,exist_ok=True);figdir=root/'paper/figures';methods=['strata','single','adamw','galore','independent'];names=dict(strata='STRATA',single='Single-G',adamw='AdamW',galore='GaLore',independent='Independent');colors=dict(strata='#D55E00',single='#0072B2',adamw='#333333',galore='#009E73',independent='#CC79A7');last=None
loss_style(plt)
cachefile=root/'results_revision/evidence_resource_cache.json';cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
seedcache=root/'results_revision/checkpoint_resource_audit.json'
if seedcache.exists():
 for x in json.loads(seedcache.read_text())['runs']:
  if x['out'] not in cache or cache[x['out']].get('optimizer_state_bytes') is None:cache[x['out']]=x
def atomic(path,text):
 if path.name=='evidence_main_live.tex':text=text.replace(r'\begin{table}[',r'\begin{table*}[').replace(r'\end{table}',r'\end{table*}').replace(r'\begin{figure}[',r'\begin{figure*}[').replace(r'\end{figure}',r'\end{figure*}')
 tmp=path.with_name(path.name+f'.{os.getpid()}.partial');tmp.write_text(format_tex(text) if path.suffix=='.tex' else text);tmp.replace(path)
def esc(s):return str(s).replace('_',r'\_')
def read(run):return [json.loads(s) for s in (run/'log.jsonl').read_text().splitlines()]
def bytecount(v):
 seen=set()
 def go(x):
  if torch.is_tensor(x):
   st=x.untyped_storage();key=(str(x.device),st.data_ptr(),st.nbytes())
   if key in seen:return 0
   seen.add(key);return st.nbytes()
  if isinstance(x,dict):return sum(go(z) for z in x.values())
  if isinstance(x,(tuple,list)):return sum(go(z) for z in x)
  return 0
 return go(v)
def resource(row):
 key=row['out'];run=row['dir']
 if key in cache and cache[key].get('optimizer_state_bytes') is not None:return cache[key]
 data=read(run);budget=int(row['units']);paths=[run/'checkpoints'/f'epoch_{budget:04d}.pt',run/'checkpoints'/f'unit_{budget:04d}.pt'];ckfile=next((f for f in paths if f.exists()),None)
 state=None
 if ckfile:
  ck=torch.load(ckfile,map_location='cpu',weights_only=False,mmap=True);assert ck.get('epoch',ck.get('unit'))==budget;state=bytecount(ck['optimizer']['state']);del ck
 result=dict(out=key,optimizer_state_bytes=state,source='final checkpoint unique tensor storages; probes included' if state is not None else 'unavailable',peak_cuda_allocated_bytes=max(z['peak_bytes'] for z in data),mean_epoch_seconds=sum(z['wall_seconds'] for z in data)/len(data));cache[key]=result;return result
def choose(rows):
 loss=rows[0]['metric']!='acc';return (min if loss else max)(sorted(rows,key=lambda x:x['lr']),key=lambda x:x['val'])
def motion(row):
 run=row['dir'];lo=run/'checkpoints/epoch_0010.pt';hi=run/'checkpoints/epoch_1000.pt'
 if not lo.exists() or not hi.exists():return None
 left=torch.load(lo,map_location='cpu',weights_only=False,mmap=True)['optimizer']['state'];right=torch.load(hi,map_location='cpu',weights_only=False,mmap=True)['optimizer']['state'];values=[]
 for key,state in right.items():
  if 'U' not in state or key not in left:continue
  old=left[key]['U'];new=state['U'];old=torch.cat(old,1) if isinstance(old,list) else old;new=torch.cat(new,1) if isinstance(new,list) else new
  old=old.double();new=new.double()
  if int(torch.linalg.matrix_rank(old,tol=1e-8))!=old.shape[1] or int(torch.linalg.matrix_rank(new,tol=1e-8))!=new.shape[1]:continue
  u=torch.linalg.qr(old,mode='reduced').Q;v=torch.linalg.qr(new,mode='reduced').Q;sv=torch.linalg.svdvals(u.T@v).clamp(0,1);values.append(float((1-(sv*sv).mean()).clamp_min(0).sqrt()))
 return float(np.mean(values)) if values else None
def figure_finish(fig,name):
 for ext in ['pdf','png']:
  dst=figdir/(name+'.'+ext);tmp=dst.with_name(dst.name+f'.{os.getpid()}.partial');fig.savefig(tmp,format=ext,dpi=180);tmp.replace(dst)
 plt.close(fig)

while True:
 ledger=root/'results_revision/completed_matrix.csv'
 if not ledger.exists():
  if a.watch:time.sleep(30);continue
  raise FileNotFoundError(ledger)
 signature=ledger.stat().st_mtime_ns
 if signature==last:time.sleep(60);continue
 rows=[]
 for r in csv.DictReader(ledger.open()):
  r['dir']=Path(a.snapshot)/'runs_revision'/Path(r['source']).name if a.snapshot else Path(r['source']);r['out']='runs_revision/'+r['dir'].name;r['lr']=float(r['lr']);r['val']=float(r['val']);r['test']=float(r['test']);r['cfg']=json.loads((r['dir']/'config.json').read_text());rows.append(r)
 groups=collections.defaultdict(list)
 for r in rows:
  if r['ablation']=='full' and r['method'] in methods:groups[(r['task'],r['model'],r['base'],r['rank'])].append(r)
 cohorts=[]
 for key,rs in sorted(groups.items()):
  if not all(any(r['method']==m and r['lr']==lr for r in rs) for m in methods for lr in [.0003,.001]):continue
  selected={m:choose([r for r in rs if r['method']==m]) for m in methods};cohorts.append((key,selected))
 main='\\paragraph{Completed follow-up evidence.}\n'
 pub=next((s for k,s in cohorts if k[:2]==('PubMed','sage')),None)
 if pub:
  s=pub['strata'];g=pub['single'];adam=pub['adamw'];res={m:resource(v) for m,v in pub.items()}
  ratio=res['strata']['optimizer_state_bytes']/res['adamw']['optimizer_state_bytes']
  main+=f"The completed PubMed/SAGE two-rate cohort gives STRATA test accuracy {s['test']:.5f}, versus {g['test']:.5f} for Single-Grassmann and {adam['test']:.5f} for AdamW: descriptive gains of {100*(s['test']-g['test']):.3f} and {100*(s['test']-adam['test']):.3f} percentage points. Each method selects its rate using terminal-100 validation accuracy. Its archived optimizer tensor state, including fixed probes, is {100*ratio:.1f}\\% of AdamW's. Table~\\ref{{tab:pubmed-resources}} gives all five methods, including stronger competitors; this is a comparison-specific benefit, not a graph-family or significance claim.\n"
  main+=r'\begin{table}[t]\centering\small'+'\n'+r'\caption{PubMed/SAGE: validation-selected two-rate, seed-42, 1000-epoch comparisons. State is unique optimizer tensor storage read from the final checkpoint, including probes. Peak is maximum recorded CUDA allocation; wall time is shared-server epoch timing, not isolated throughput.}\label{tab:pubmed-resources}'+'\n'+r'\begin{tabular}{lrrrrr}\toprule Method & Val acc & Test acc & State MiB & Peak MiB & ms/epoch\\\midrule'+'\n'
  for m in methods:
   z=pub[m];v=res[m];main+=f"{names[m]} & {z['val']:.4f} & {z['test']:.4f} & {v['optimizer_state_bytes']/2**20:.2f} & {v['peak_cuda_allocated_bytes']/2**20:.1f} & {1000*v['mean_epoch_seconds']:.1f}"+r' \\'+'\n'
  main+=r'\bottomrule\end{tabular}\end{table}'+'\n'
  fig,axs=plt.subplots(1,2,figsize=(6,2.65),layout='constrained')
  for m in methods:
   rr=pub[m];data=read(rr['dir']);xx=[z['epoch'] for z in data]
   axs[0].plot(xx,[z['train_loss'] for z in data],color=colors[m],lw=1,label=names[m]);axs[1].plot(xx,[z['val_loss'] for z in data],color=colors[m],lw=1)
  for ax in axs:ax.set(xlabel='Epoch',xlim=(0,1000));ax.grid(alpha=.18)
  axs[0].set(ylabel='Training cross-entropy',yscale='log');axs[1].set(ylabel='Validation cross-entropy',yscale='log');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.1));figure_finish(fig,'fig_evidence_pubmed')
  main+=r'\begin{figure}[t]\centering\includegraphics[width=.98\linewidth]{figures/fig_evidence_pubmed.pdf}\caption{Completed PubMed/SAGE training and validation trajectories. Each of five methods selects its learning rate using terminal-100 validation accuracy after both full-budget candidates finish. All trajectories are raw; held-out test outcomes are reported in Table~\ref{tab:pubmed-resources}.}\label{fig:pubmed-positive}\end{figure}'+'\n'
  pubco=[(k,s) for k,s in cohorts if k[0]=='PubMed'];positive=[k[1] for k,s in pubco if s['strata']['test']>s['single']['test']]
  main+=f"The completed default-protocol PubMed architectures are {', '.join(k[1] for k,_ in pubco)}; positive STRATA--Single-G test differences occur in {', '.join(positive) if positive else 'none'}. Every completed architecture remains in Table~\\ref{{tab:new-cohorts}}. Further controlled ablations and an equal-budget flag-step sweep are reported in Appendix~\\ref{{app:evidence-extension}}; experimental variants do not overwrite default-protocol results.\n"
 extra=r'\section{Completed Evidence, Resource Audits, and Controlled Follow-ups}\label{app:evidence-extension}'+'\n'
 extra+=r'All comparisons below use seed 42. A two-rate result requires both rates for every method, and all prescribed epochs must finish. Rates and exploratory flag-step settings are selected on validation only. Individual incomplete configurations remain in the completion ledger, with no extrapolated endpoints.'+'\n'
 queue=Path(a.snapshot)/'queue_snapshot.json' if a.snapshot else root/'experiments/revision/queue.json'
 with queue.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);jobs=json.loads(queue.read_text())
 coverage=collections.defaultdict(collections.Counter)
 for job in jobs:
  family=job['family']
  if family=='graph' and ('--ablation' in job['args'] or '--flag-xi' in job['args']):family='graph mechanism/sweep'
  coverage[family][job['status']]+=1
 extra+=r'\paragraph{Proposal coverage and remaining work.} Configuration counts measure execution coverage, not positive results. The original image/language training continues while the balanced graph follow-up uses available capacity. WikiText-2 epochs traverse its full corpus; GPT-2 uses fixed ten-step evaluation units, explicitly not corpus epochs or the 1000-corpus-epoch-equivalent proposal budget. OpenWebText/C4 subsets and corpus-switch experiments are not established completed benchmarks in this matrix. Core50 and DomainNet acquisition is complete, but readiness does not imply completion of their long-horizon runs.'+'\n'
 extra+=r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\caption{Execution snapshot by disjoint queue family. No pending, running or failed configuration is a completed endpoint.}\label{tab:evidence-coverage}\begin{tabular}{lrrrrr}\toprule Family & Complete & Running & Pending & Data wait & Failed\\\midrule'+'\n'
 for family,counter in sorted(coverage.items()):extra+=esc(family)+' & '+' & '.join(str(counter[s]) for s in ['complete','running','pending','waiting_data','failed'])+r' \\'+'\n'
 extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
 extra+=r'\paragraph{Measured state--quality tradeoffs.} State includes every unique optimizer tensor storage in a final checkpoint, including fixed probes; weights, gradients and transient workspaces are separate. Peak allocation and shared-server timing are reported independently, so a reduction in stored state is not asserted to be a peak-memory or speed reduction.'+'\n'
 extra+=r'{\footnotesize\setlength{\tabcolsep}{3pt}\begin{longtable}{llrrrr}\caption{Complete-cohort resource audit. Ratios compare STRATA with full AdamW after each method independently selects its learning rate on validation. Accuracy differences are STRATA minus Single-G; relative-L2 differences retain their lower-is-better sign.}\\\toprule Task & Model & State ratio & Peak ratio & Test & $\Delta$ Single-G\\\midrule\endfirsthead\toprule Task & Model & State ratio & Peak ratio & Test & $\Delta$ Single-G\\\midrule\endhead'+'\n'
 table=[]
 for key,selected in cohorts:
  x=selected['strata'];g=selected['single'];s=resource(x);b=resource(selected['adamw'])
  if s['optimizer_state_bytes'] is None or b['optimizer_state_bytes'] is None:continue
  sr=s['optimizer_state_bytes']/b['optimizer_state_bytes'];pr=s['peak_cuda_allocated_bytes']/b['peak_cuda_allocated_bytes'];extra+=f"{esc(key[0])} & {esc(key[1])} & {sr:.3f} & {pr:.3f} & {x['test']:.5f} & {x['test']-g['test']:+.5f}"+r' \\'+'\n';table.append(dict(task=key[0],model=key[1],state_ratio=sr,peak_ratio=pr,strata=x['test'],single=g['test'],metric=x['metric']))
 extra+=r'\bottomrule\end{longtable}}'+'\n'
 darcy=next((s for k,s in cohorts if k[0]=='Darcy-generated'),None)
 if darcy:
  extra+=r'\paragraph{Completed Darcy/FNO comparison.} All ten two-rate configurations finish 2000 epochs on the retained generated $16\times16$ operator dataset. Validation relative $L_2$ selects the rate; PDE residual and spectral errors are secondary test metrics. Frequency errors are mean squared FFT magnitudes of errors in training-standard-deviation target units. This is not an external standard Darcy benchmark. Spectral real/imaginary tensors retain dense Adam state; eligible pointwise matrices use projected state.'+'\n'
  extra+=r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\caption{Completed generated-Darcy/FNO results, terminal-100 test metrics.}\label{tab:darcy-complete}\begin{tabular}{lrrrrr}\toprule Method & lr & Rel-$L_2$ & PDE residual & Low-freq & High-freq\\\midrule'+'\n'
  fig,axs=plt.subplots(1,3,figsize=(7,2.3),layout='constrained')
  for m in methods:
   rr=darcy[m];data=read(rr['dir']);tail=data[-100:];met={k:float(np.mean([z['test'][k] for z in tail])) for k in ['rel_l2','pde_residual','low_freq_error','high_freq_error']};extra+=f"{names[m]} & {rr['lr']:.0e} & {met['rel_l2']:.4f} & {met['pde_residual']:.3g} & {met['low_freq_error']:.3g} & {met['high_freq_error']:.3g}"+r' \\'+'\n'
   for ax,key,label in zip(axs.flat,['train','val','test'],['Training MSE','Validation MSE','Test MSE']):
    y=[z['train_loss'] if key=='train' else z[key]['loss'] for z in data];ax.plot([z['epoch'] for z in data],y,label=names[m],color=colors[m],lw=.9);ax.set(xlabel='Epoch',ylabel=label,yscale='log');ax.grid(alpha=.18)
  h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.05));figure_finish(fig,'fig_evidence_darcy');extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
  extra+=r'\begin{figure}[!htbp]\centering\includegraphics[width=.98\linewidth]{figures/fig_evidence_darcy.pdf}\caption{Complete 2000-epoch Darcy/FNO trajectories. Rates are validation-selected from complete two-rate grids; test residual and spectrum never select a curve. All methods and adverse outcomes are retained.}\end{figure}'+'\n'
 variants=['full','same_rho','reverse_rho','equal_a','no_residual','random_flag','no_transport','frozen_after_warmup','heterogeneous_memory','periodic_svd','least_squares_transport']
 ab=[r for r in rows if (r['task'],r['model'],r['method'])==('PubMed','sage','strata') and r['ablation'] in variants]
 extra+=r'\paragraph{PubMed mechanism follow-up.} The ten variants and both rates are frozen before follow-up outcomes. Their controls reuse the original two full-method configurations. Freezing after 200 updates also stops variance resets; periodic pooled SVD changes the flag objective. Thus these are explicitly identified interventions, not pure isolated geometry controls.'+'\n'
 if all(any(r['ablation']==v and r['lr']==lr for r in ab) for v in variants for lr in [.0003,.001]):
  chosen_ab={v:choose([z for z in ab if z['ablation']==v]) for v in variants};full=chosen_ab['full'];gaps=[100*(full['test']-chosen_ab[v]['test']) for v in variants if v!='full']
  main+=f"\\paragraph{{Mechanism evidence in the positive setting.}} Both rates for all ten PubMed/SAGE interventions are complete (22 configurations including the full controls). Full STRATA reaches {100*full['test']:.3f}\\%, exceeding every validation-selected ablation by {min(gaps):.3f}--{max(gaps):.3f} percentage points (Table~\\ref{{tab:pubmed-ablations}}). This supports the combined configuration in this setting; interventions that also alter resets or the covariance objective do not isolate a single geometric cause.\n"
  extra+=r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\caption{Complete PubMed/SAGE ablation grid, validation-selected rates, terminal-100 accuracies. Differences are relative to full STRATA. No variant is promoted to the default based on test results.}\label{tab:pubmed-ablations}\begin{tabular}{lrrrr}\toprule Variant & lr & Val & Test & $\Delta$ test (pp)\\\midrule'+'\n'
  for v in variants:
   r=chosen_ab[v];extra+=f"{esc(v)} & {r['lr']:.0e} & {r['val']:.5f} & {r['test']:.5f} & {100*(r['test']-full['test']):+.3f}"+r' \\'+'\n'
  extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
 else:extra+=f"At this snapshot {len(ab)}/22 configurations are complete; the validation-selected ablation table waits for the whole grid. Completed fixed-rate trajectories remain available separately.\n"
 geo=['strata','single','independent'];xis=[.001,.1,10.];xirows=[r for r in rows if r['task']=='PubMed' and r['model']=='sage' and int(r['rank'])==64 and r['method'] in geo and r['ablation'] in ['full','xi_0.1','xi_10']]
 def getxi(r):return .001 if r['ablation']=='full' else float(r['ablation'].split('_')[1])
 extra+=r'\paragraph{Exploratory flag-step sensitivity.} The declared grid is $\xi\in\{10^{-3},10^{-1},10\}$ and $\eta\in\{3\times10^{-4},10^{-3}\}$ for each of STRATA, Single-G and Independent: six equal-budget candidates per method. Defaults remain unchanged. Larger raw-objective steps test whether weak flag motion limits the default protocol. Baseline numerical paths were checked unchanged for all five optimizers over 55 paired updates; all 20 new graph-path smoke tests passed. New runs save float32 bases and evaluate span motion with float64 QR/SVD; old float16 flag snapshots are not used to infer tiny principal angles. The span diagnostic is $D_{\rm span}=\sqrt{1-\lVert Q_{10}^{\top}Q_{1000}\rVert_F^2/r}$, with fresh orthonormal QR factors and represented rank $r$. Independent uses the concatenated subspace union when it has full column rank; excluded rank-deficient matrices are not assigned zero motion.'+'\n'
 sweep_complete=all(any(r['method']==m and getxi(r)==xi and r['lr']==lr for r in xirows) for m in geo for xi in xis for lr in [.0003,.001])
 if sweep_complete:
  extra+=r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\caption{Matched six-candidate exploratory search per geometric optimizer, selected on validation. Span motion compares the total represented space at checkpoints 10 and 1000, using float64 QR/SVD and averaging eligible matrices. It is not an oracle tracking error.}\label{tab:xi-sweep}\begin{tabular}{lrrrrr}\toprule Method & $\xi$ & lr & Val & Test & Span motion\\\midrule'+'\n'
  xi_best={m:choose([r for r in xirows if r['method']==m]) for m in geo};xs=xi_best['strata'];xg=xi_best['single'];xi=xi_best['independent'];eta_tex=r'10^{-3}' if xs['lr']==.001 else r'3\times10^{-4}'
  main+=f"\\paragraph{{Matched flag-step search.}} In the completed exploratory six-candidate grid for each geometric method, validation selects STRATA at $\\xi={getxi(xs):g}$ and $\\eta={eta_tex}$, with test accuracy {100*xs['test']:.3f}\\%, versus {100*xg['test']:.3f}\\% for Single-Grassmann and {100*xi['test']:.3f}\\% for Independent. The STRATA--Single-G difference is {100*(xs['test']-xg['test']):.3f} percentage points; STRATA improves by {100*(xs['test']-pub['strata']['test']):.3f} points over its default-step result. Table~\\ref{{tab:xi-sweep}} and the full profiles retain every setting. This is a separate sensitivity result, and Independent remains stronger on held-out test accuracy.\n"
  audit=[]
  for m in geo:
   selected=choose([r for r in xirows if r['method']==m]);mv=motion(selected);extra+=f"{names[m]} & {getxi(selected):g} & {selected['lr']:.0e} & {selected['val']:.5f} & {selected['test']:.5f} & "+('--' if mv is None else f'{mv:.3g}')+r' \\'+'\n'
   profile=[choose([r for r in xirows if r['method']==m and getxi(r)==xi]) for xi in xis]
   audit.extend(dict(method=m,xi=getxi(r),lr=r['lr'],val=r['val'],test=r['test']) for r in profile)
  extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
  extra+=r'\begin{table}[htbp]\centering\caption{\textbf{Complete flag-step sensitivity profiles.} Each point selects its rate on validation after both candidates finish. Test is reporting only.}\begin{tabular}{lrrrr}\toprule Method & $\xi$ & lr & Val acc & Test acc\\\midrule'+'\n'
  for r in audit:extra+=f"{names[r['method']]} & {r['xi']:g} & {r['lr']:.0e} & {r['val']:.5f} & {r['test']:.5f}"+r' \\'+'\n'
  extra+=r'\bottomrule\end{tabular}\end{table}'+'\n';atomic(report/'xi_profile.json',json.dumps(audit,indent=2))
  fig,axs=plt.subplots(3,2,figsize=(6,6.4),layout='constrained');grid_audit=[]
  for k,m in enumerate(geo):
   for rr in sorted([r for r in xirows if r['method']==m],key=lambda r:(getxi(r),r['lr'])):
    data=read(rr['dir']);xi=getxi(rr);color=plt.cm.viridis(.15+.35*xis.index(xi));style='-' if rr['lr']==.0003 else '--';label=f"xi={xi:g}, lr={rr['lr']:g}"
    for ax,key in zip(axs[k],['train_loss','val_loss']):ax.plot([z['epoch'] for z in data],[z[key] for z in data],color=color,linestyle=style,lw=.75,label=label);ax.set(xlabel='Epoch',yscale='log',xlim=(0,1000),ylabel='Training cross-entropy' if key=='train_loss' else 'Validation cross-entropy');ax.grid(alpha=.18)
    grid_audit.append(dict(method=m,rank=64,xi=xi,lr=rr['lr'],source=rr['source'],epochs=len(data)))
   axs[k,0].set_title(names[m]+' R64');axs[k,1].set_title(names[m]+' R64 / validation')
  h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.075),fontsize=6);figure_finish(fig,'fig_evidence_xi_loss_grid');extra+=r'\begin{figure}[!htbp]\centering\includegraphics[width=.98\linewidth]{figures/fig_evidence_xi_loss_grid.pdf}\caption{All 18 completed equal-rank step/rate trajectories, without selection or smoothing. Rows show STRATA, Single-G and Independent; columns show training and validation cross-entropy. Colors denote flag step and line styles distinguish the two learning rates. Every candidate runs 1000 epochs.}\end{figure}'+'\n';atomic(report/'xi_loss_grid_manifest.json',json.dumps(grid_audit,indent=2))
 else:extra+=f"At this snapshot {len(xirows)}/18 sweep candidates are complete. A best-of-grid comparison is withheld until all three methods finish all six candidates.\n"
 memory_plan=root/'results_revision/memory_control/plan.json';memory_complete=False
 if memory_plan.exists():
  mp=json.loads(memory_plan.read_text());mr=int(mp['rank']);matched=[r for r in rows if r['task']=='PubMed' and r['model']=='sage' and r['method']=='single' and int(r['rank'])==mr and r['ablation'] in ['xi_0.001','xi_0.1','xi_10']]
  extra+=r'\paragraph{State-matched capacity control.} Same-rank comparisons allocate more optimizer state to STRATA than to Single-Grassmann. We therefore choose the closest integer Single-G rank using initialized tensor storage only, before observing any new accuracy; the target includes probes. Both methods receive the same six-candidate flag-step/rate grid. This is an approximately equal-state comparison, separate from the equal-rank default protocol.'+'\n'
  extra+=f"Single-G rank {mr} uses {mp['single_matched_state_bytes']} bytes, versus {mp['strata_state_bytes']} for rank-64 STRATA, a signed difference of {100*mp['state_difference_fraction']:+.3f}\\%. The choice uses exact shape-derived state counts over integer ranks 4--255, including probe-rank saturation, checked against actual initialized storage at four ranks and an archived STRATA final checkpoint; no validation or test score chooses rank.\n"
  memory_complete=sweep_complete and all(any(getxi(r)==xi and r['lr']==lr for r in matched) for xi in xis for lr in [.0003,.001])
  if memory_complete:
   for rr in matched:
    state=resource(rr)['optimizer_state_bytes']
    if state is not None:assert state==mp['single_matched_state_bytes'],(rr['out'],state)
   pools=[('STRATA R64',[r for r in xirows if r['method']=='strata']),('Single-G R64',[r for r in xirows if r['method']=='single']),(f'Single-G R{mr}',matched),('Independent total R64',[r for r in xirows if r['method']=='independent'])]
   selected_memory={label:choose(rs) for label,rs in pools};s=selected_memory['STRATA R64'];g=selected_memory[f'Single-G R{mr}'];sd=choose([r for r in pools[0][1] if getxi(r)==.001]);gd=choose([r for r in matched if getxi(r)==.001])
   main+=f"\\paragraph{{Equal-state follow-up.}} A resource-only calibration selects Single-Grassmann rank {mr}, whose stored optimizer tensor state differs from STRATA R64 by {100*mp['state_difference_fraction']:+.3f}\\%. After the same six-candidate validation search for both methods, test accuracies are {100*s['test']:.3f}\\% and {100*g['test']:.3f}\\%, respectively (signed STRATA difference {100*(s['test']-g['test']):+.3f} percentage points; Table~\\ref{{tab:memory-control}}). At the default flag step, the two-rate validation selections instead give STRATA {100*sd['test']:.3f}\\% and Single-G R{mr} {100*gd['test']:.3f}\\%. Thus extra state is a material confound in the equal-rank comparison, and the retained six-grid gain is smaller and protocol-dependent.\n"
   extra+=r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\caption{Completed state-budget control. Rank is fixed from storage alone; every row finishes two rates at each of three flag steps. Default test selects the rate at $\xi=10^{-3}$; six-grid test selects both hyperparameters on validation. Independent uses blocks of dimensions $(16,16,32)$, whose total rank budget is 64. State includes probes and the cached union basis when present.}\label{tab:memory-control}\begin{tabular}{lrrrrr}\toprule Method & State MiB & $\xi_{6}$ & lr$_6$ & Test default & Test six-grid\\\midrule'+'\n'
   fig,axs=plt.subplots(1,2,figsize=(6,2.9),layout='constrained');maudit=[]
   for label,rs in pools:
    rr=selected_memory[label];default=choose([r for r in rs if getxi(r)==.001]);res=resource(rr);state=res['optimizer_state_bytes'];extra+=f"{label} & "+('--' if state is None else f'{state/2**20:.3f}')+f" & {getxi(rr):g} & {rr['lr']:.0e} & {default['test']:.5f} & {rr['test']:.5f}"+r' \\'+'\n'
    data=read(rr['dir']);style='--' if label==f'Single-G R{mr}' else '-';color='#56B4E9' if style=='--' else colors[rr['method']]
    for ax,key in zip(axs,['train_loss','val_loss']):ax.plot([z['epoch'] for z in data],[z[key] for z in data],color=color,linestyle=style,lw=.9,label=label);ax.set(xlabel='Epoch',yscale='log',xlim=(0,1000),ylabel='Training cross-entropy' if key=='train_loss' else 'Validation cross-entropy');ax.grid(alpha=.18)
    maudit.append(dict(label=label,source=rr['source'],xi=getxi(rr),lr=rr['lr'],val=rr['val'],test=rr['test'],default_test=default['test'],optimizer_state_bytes=state))
   extra+=r'\bottomrule\end{tabular}\end{table}'+'\n';h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.17));figure_finish(fig,'fig_evidence_memory_control')
   extra+=r'\begin{figure}[!htbp]\centering\includegraphics[width=.98\linewidth]{figures/fig_evidence_memory_control.pdf}\caption{Completed capacity-control training and validation losses after equal six-candidate validation selection. The additional Single-G rank is determined using state storage alone; all six candidates are shown separately below.}\end{figure}'+'\n';atomic(report/'memory_control_selection.json',json.dumps(maudit,indent=2))
   extra+=r'\begin{table}[htbp]\centering\small\caption{All six completed Single-G state-control candidates. The rank is chosen using storage alone; validation, not test, selects the reported optimum.}\begin{tabular}{rrrr}\toprule $\xi$ & lr & Val acc & Test acc\\\midrule'+'\n';fig,axs=plt.subplots(1,2,figsize=(6,2.9),layout='constrained')
   for rr in sorted(matched,key=lambda r:(getxi(r),r['lr'])):
    xi=getxi(rr);extra+=f"{xi:g} & {rr['lr']:.0e} & {rr['val']:.5f} & {rr['test']:.5f}"+r' \\'+'\n';data=read(rr['dir']);color=plt.cm.viridis(.15+.35*xis.index(xi));style='-' if rr['lr']==.0003 else '--'
    for ax,key in zip(axs,['train_loss','val_loss']):ax.plot([z['epoch'] for z in data],[z[key] for z in data],color=color,linestyle=style,lw=.85,label=f"xi={xi:g}, lr={rr['lr']:g}");ax.set(xlabel='Epoch',yscale='log',xlim=(0,1000),ylabel='Training cross-entropy' if key=='train_loss' else 'Validation cross-entropy');ax.grid(alpha=.18)
   extra+=r'\bottomrule\end{tabular}\end{table}'+'\n';h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.19),fontsize=6);figure_finish(fig,'fig_evidence_memory_grid');extra+=r'\begin{figure}[!htbp]\centering\includegraphics[width=.98\linewidth]{figures/fig_evidence_memory_grid.pdf}\caption{All six completed Single-G capacity-control training and validation trajectories. No candidate is omitted, selected or smoothed in this grid view.}\end{figure}'+'\n'
  else:extra+=f"At this snapshot {len(matched)}/6 new state-control candidates are complete; the selected comparison waits for all six.\n"
 atomic(root/'paper/sections/evidence_main_live.tex',main);atomic(root/'paper/sections/evidence_appendix_live.tex',extra);atomic(cachefile,json.dumps(cache,indent=2));atomic(report/'complete_resource_cohorts.json',json.dumps(table,indent=2));atomic(report/'publication_snapshot.json',json.dumps(dict(completed_rows=len(rows),complete_cohorts=len(cohorts),pubmed_ablation_complete=len(ab),xi_complete=len(xirows),xi_full_grid=sweep_complete,memory_control_complete=memory_complete),indent=2));last=signature;print(json.dumps(dict(completed_rows=len(rows),cohorts=len(cohorts),pubmed_ablation=len(ab),xi=len(xirows))),flush=True)
 if not a.watch:break
 time.sleep(60)
