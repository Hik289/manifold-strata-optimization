"""Publish all prospective settings; require exhausted equal-count searches."""
import argparse,collections,fcntl,hashlib,json,math,os,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_table_style import format_tex,loss_style
from paper_new_family_summary import build_summary
p=argparse.ArgumentParser();p.add_argument('--root',default='..');p.add_argument('--snapshot');p.add_argument('--watch',action='store_true');a=p.parse_args();root=Path(a.root).resolve();folder=root/'results_revision/new_families';report=root/'analysis/revision_20261102/new_families';report.mkdir(parents=True,exist_ok=True);methods=['strata','single','adamw','galore','independent'];names=dict(strata='STRATA',single='Single-G',adamw='AdamW',galore='GaLore',independent='Independent');colors=dict(strata='#D55E00',single='#0072B2',adamw='#333333',galore='#009E73',independent='#CC79A7');last=None
markers=dict(strata='o',single='s',adamw='^',galore='D',independent='P')
loss_style(plt)
def atomic(path,text):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.partial');tmp.write_text(format_tex(text) if path.suffix=='.tex' else text);tmp.replace(path)
def escape(x):return str(x).replace('_',r'\_')
def display(task,base):
 name={'gas_drift':'Gas sensors','ml100k':'MovieLens100K','ml1m':'MovieLens1M'}.get(task,task)
 return name+(' / fine-tuning' if base=='finetune' else ' / replay' if base=='replay' else ' / H24' if task.startswith('ETT') else ' / ratings')
def finish(fig,name):
 for ext in ['pdf','png']:
  out=root/'paper/figures'/(name+'.'+ext);tmp=out.with_name(out.name+f'.{os.getpid()}.partial');fig.savefig(tmp,format=ext,dpi=170);tmp.replace(out)
 plt.close(fig)
while True:
 if not (folder/'plan.json').exists():
  if not a.watch:raise FileNotFoundError(folder/'plan.json')
  time.sleep(30);continue
 plan=json.loads((folder/'plan.json').read_text());q=Path(a.snapshot)/'queue_snapshot.json' if a.snapshot else root/'experiments/revision/queue.json'
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);jobs=[j for j in json.loads(q.read_text()) if j.get('new_family_extension')]
 signature=tuple((j['id'],j['status']) for j in jobs)
 if a.watch and signature==last:time.sleep(30);continue
 coverage=collections.defaultdict(collections.Counter);groups=collections.defaultdict(list);candidates=[]
 for j in jobs:
  coverage[j['family']][j['status']]+=1;args=j['args'];task=args[args.index('--task')+1];base=args[args.index('--base')+1];groups[(j['family'],task,base)].append(j)
 complete=[];tex=r'\subsection{Prospective new-task execution and complete comparisons}\label{app:new-family-results}'+'\n'
 tex+=r'This extension was declared before its outcomes: six settings, five methods and six candidates per method. Every formal run uses seed 42 and 1000 complete epochs. Configuration selection uses terminal-100 validation metrics. A comparison appears only after all 30 declared candidates have completed or recorded failure; failures have no fabricated endpoint. A method with no complete candidate has no selected score. Negative outcomes and every declared setting remain in this record.'+'\n'
 tex+=r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\caption{Execution coverage of every declared new setting. Counts are configurations, not quality wins; failed candidates remain visible.}\begin{tabular}{llrrrr}\toprule Family & Setting & Complete & Running & Pending & Failed\\\midrule'+'\n'
 for key,js in sorted(groups.items()):
  c=collections.Counter(j['status'] for j in js);tex+=escape(key[0])+' & '+escape(display(key[1],key[2]))+' & '+' & '.join(str(c[s]) for s in ['complete','running','pending','failed'])+r' \\'+'\n'
 tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'+r'\FloatBarrier'+'\n'
 for key,js in sorted(groups.items()):
  if len(js)!=30 or any(j['status'] not in ['complete','failed'] for j in js):continue
  rows=[];failed=collections.Counter()
  for j in js:
   args=j['args'];method=args[args.index('--method')+1]
   if j['status']=='failed':failed[method]+=1;continue
   run=Path(a.snapshot)/'runs_revision'/Path(j['out']).name if a.snapshot else root/'experiments'/j['out'];cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text());data=[json.loads(s) for s in (run/'log.jsonl').read_text().splitlines()];assert summary['complete'] and not summary.get('smoke') and len(data)==1000;tail=data[-100:];rows.append(dict(id=j['id'],out=j['out'],method=method,lr=float(cfg['lr']),xi=cfg.get('flag_xi'),val=sum(z['val_metric'] for z in tail)/100,test=sum(z['test_metric'] for z in tail)/100,state_bytes=data[-1]['optimizer_state_bytes'],metric=cfg['primary_metric'],direction=cfg['primary_direction'],data=data))
  candidates.extend([{k:v for k,v in x.items() if k!='data'} for x in rows])
  if any(not any(z['method']==m for z in rows) for m in methods):continue
  lower=rows[0]['direction']=='min';selected={m:(min if lower else max)(sorted([z for z in rows if z['method']==m],key=lambda z:(z['lr'],z['xi'] or 0)),key=lambda z:z['val']) for m in methods};complete.append(dict(key=key,selected=selected,failed=dict(failed),lower=lower))
 if complete:
  tex+=r'{\footnotesize\setlength{\tabcolsep}{3pt}\begin{longtable}{llrrrrr}\caption{All fully searched new settings. Each displayed method independently selects among six candidates on validation. Values are terminal-100 test metrics; MSE and RMSE are lower-is-better, accuracy is higher-is-better. No setting is omitted according to outcome.}\label{tab:new-families-complete}\\\toprule Setting & Metric & STRATA & Single-G & AdamW & GaLore & Indep.\\\midrule\endfirsthead\toprule Setting & Metric & STRATA & Single-G & AdamW & GaLore & Indep.\\\midrule\endhead'+'\n'
  for c in complete:
   fam,task,base=c['key'];s=c['selected'];best=(min if c['lower'] else max)(s[m]['test'] for m in methods);tex+=escape(display(task,base))+' & '+s['strata']['metric']+' & '+' & '.join((r'\textbf{'+f"{s[m]['test']:.5f}"+'}') if s[m]['test']==best else f"{s[m]['test']:.5f}" for m in methods)+r' \\'+'\n'
  tex+=r'\bottomrule\end{longtable}}'+'\n'
 else:tex+='No setting yet supplies a complete five-method comparison after the declared search. Partial runs, all-failed method searches and smoke checks do not receive fabricated endpoint comparisons.\n'
 audits=[];family_scope=collections.defaultdict(lambda:dict(settings=0,strata_better_than_single=0,strata_better_than_adamw=0,strata_on_pareto_front=0))
 for c in complete:
  tex+=r'\clearpage'+'\n'
  fam,task,base=c['key'];s=c['selected'];lower=c['lower'];better=lambda x,y:x<y if lower else x>y;eqbetter=lambda x,y:x<=y if lower else x>=y;pareto={m:not any(s[n]['state_bytes']<=s[m]['state_bytes'] and eqbetter(s[n]['test'],s[m]['test']) and (s[n]['state_bytes']<s[m]['state_bytes'] or better(s[n]['test'],s[m]['test'])) for n in methods if n!=m) for m in methods};scope=family_scope[fam];scope['settings']+=1;scope['strata_better_than_single']+=int(better(s['strata']['test'],s['single']['test']));scope['strata_better_than_adamw']+=int(better(s['strata']['test'],s['adamw']['test']));scope['strata_on_pareto_front']+=int(pareto['strata']);label=display(task,base);tex+='\\paragraph{'+escape(label)+' complete search.} '+f"Primary metric: {s['strata']['metric']}. Recorded candidate failures: "+', '.join(f'{names[m]} {c["failed"].get(m,0)}' for m in methods)+'. Shared-CPU wall time is not isolated throughput; tensor-state size is distinct from peak memory.\n'
  tex+=r'\begin{table}[htbp]\centering\caption{\textbf{'+escape(label)+r': selected configurations and stored state.} Each method selects among six candidates on validation. Bold marks the best validation/test value; the green row identifies STRATA. Pareto status uses test quality and stored state, not peak memory.}\begin{tabular}{lrrrrrl}\toprule Method & lr & $\xi$ & Val & Test & State MiB & Pareto\\\midrule'+'\n'
  bestval=(min if lower else max)(s[m]['val'] for m in methods);besttest=(min if lower else max)(s[m]['test'] for m in methods)
  for m in methods:
   r=s[m];v=f"{r['val']:.5f}";t=f"{r['test']:.5f}";v=r'\textbf{'+v+'}' if r['val']==bestval else v;t=r'\textbf{'+t+'}' if r['test']==besttest else t;tex+=f"{names[m]} & {r['lr']:.0e} & "+('--' if r['xi'] is None else f"{r['xi']:g}")+f" & {v} & {t} & {r['state_bytes']/2**20:.3f} & "+(r'\rwcheck' if pareto[m] else r'\rwcross')+r' \\'+'\n'
  tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
  fig,axs=plt.subplots(1,3,figsize=(7,2.3),layout='constrained')
  for m in methods:
   r=s[m];data=r['data'];xx=[z['epoch'] for z in data]
   for ax,key in zip(axs,['train_loss','val_loss','test_loss']):ax.plot(xx,[z[key] for z in data],label=names[m],color=colors[m],lw=.9,marker=markers[m],markevery=200,markersize=2.5)
   audits.append(dict(setting=label,setting_id=task+'/'+base,family=fam,method=m,lr=r['lr'],xi=r['xi'],val=r['val'],test=r['test'],state_bytes=r['state_bytes'],pareto=pareto[m],failed_candidates=c['failed'].get(m,0),source=r['out']))
  for ax,label_loss in zip(axs,['Training loss','Validation loss','Test loss']):ax.set(xlabel='Epoch',ylabel=label_loss,yscale='log');ax.grid(alpha=.18)
  h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.12));name='fig_new_family_'+hashlib.sha256(label.encode()).hexdigest()[:10];finish(fig,name);tex+='\\begin{figure}[!htbp]\\centering\\includegraphics[width=.98\\linewidth]{figures/'+name+'.pdf}\\caption{Complete '+escape(label)+r' loss trajectories. Left to right: training, validation, and test loss. Selection uses validation only after all candidates finish or record failure. All five methods and adverse outcomes remain visible. Curves are raw and unsmoothed.'+(' Sensor-drift evaluation averages domains seen so far and changes at domain boundaries.' if task=='gas_drift' else '')+r'}\end{figure}'+'\n'
 main=build_summary(complete,methods,display)
 atomic(root/'paper/sections/new_families_summary_live.tex',main)
 snapshot_time=json.loads((root/'results_revision/progress_snapshot.json').read_text())['timestamp_utc'].replace('T',' ')[:16]+' UTC' if a.snapshot else time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime())
 atomic(root/'paper/sections/execution_snapshot_live.tex',r'\def\experimentSnapshot{'+snapshot_time+'}\n')
 atomic(report/'execution_audit.json',json.dumps([dict(id=j['id'],family=j['family'],status=j['status'],out=j['out'],args=j['args'],returncode=j.get('returncode')) for j in jobs],indent=2))
 atomic(root/'paper/sections/new_families_results_live.tex',tex);atomic(report/'selection_audit.json',json.dumps(audits,indent=2));atomic(report/'candidate_audit.json',json.dumps(candidates,indent=2));snapshot=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),data_snapshot_utc=json.loads((root/'results_revision/progress_snapshot.json').read_text())['timestamp_utc'] if a.snapshot else None,data_mode='frozen local snapshot' if a.snapshot else 'live training host queue',counts=dict(collections.Counter(j['status'] for j in jobs)),complete_settings=len(complete),family_scope=dict(family_scope),declared_settings=6,declared_configurations=180,selection='terminal100 validation; failed candidates retained; no post-outcome deletion');atomic(report/'publication_snapshot.json',json.dumps(snapshot,indent=2));last=signature;print(json.dumps(snapshot),flush=True)
 if not a.watch:break
 time.sleep(30)
