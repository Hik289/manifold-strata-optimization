"""Keep all completed ablation trajectories; never select a variant by test loss."""
import argparse,collections,csv,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',default='..');p.add_argument('--snapshot');p.add_argument('--watch',action='store_true');a=p.parse_args();root=Path(a.root).resolve();last=None
labels={'same_rho':'Equal covariance EMA','reverse_rho':'Reverse covariance EMA','equal_a':'Equal coefficients','no_residual':'No residual','random_flag':'Fixed random flag','no_transport':'No coordinate rotation','frozen_after_warmup':'Frozen learned flag','heterogeneous_memory':'Heterogeneous moments','periodic_svd':'Periodic pooled SVD','least_squares_transport':'Least-squares transport'}
plt.rcParams.update({'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
while True:
 ledger=root/'results_revision/completed_matrix.csv'
 if not ledger.exists():
  if a.watch:time.sleep(60);continue
  raise FileNotFoundError(ledger)
 sig=ledger.stat().st_mtime_ns
 if a.watch and sig==last:time.sleep(60);continue
 baselines={};variants=collections.defaultdict(list)
 for row in csv.DictReader(ledger.open()):
  if row['method']!='strata':continue
  key=(row['task'],row['model'],row['rank'],row['lr']);directory=Path(row['source'])
  if a.snapshot:directory=Path(a.snapshot)/'runs_revision'/directory.name
  if not (directory/'log.jsonl').exists():continue
  row['dir']=directory
  if row['ablation']=='full':baselines[key]=row
  elif row['ablation'] in labels:variants[key].append(row)
 tex='';audit=[]
 for key,rows in sorted(variants.items()):
  if key not in baselines:continue
  if not tex:tex='\\subsection{Completed ablation loss trajectories}\nEach panel uses one predeclared fixed learning rate and only full-budget completed runs. The ablation matrix may still be incomplete; missing variants are not assigned values. All completed variants, including adverse ones, are retained.\n'
  fig,axes=plt.subplots(1,2,figsize=(6,2.5),layout='constrained');rows=[baselines[key]]+sorted(rows,key=lambda x:x['ablation'])
  for k,row in enumerate(rows):
   data=[json.loads(s) for s in (row['dir']/'log.jsonl').read_text().splitlines()];xx=[r['epoch'] for r in data];vl=[r.get('val_loss',np.mean([z['loss'] for z in r.get('val_per_task',[])]) if r.get('val_per_task') else float('nan')) for r in data];label='Full STRATA' if k==0 else labels[row['ablation']];color='black' if k==0 else plt.cm.tab20((k-1)%20)
   for axis,yy in zip(axes,[[r['train_loss'] for r in data],vl]):axis.plot(xx,yy,label=label,color=color,lw=1.2 if k==0 else .85)
   audit.append(dict(cohort=key,ablation=row['ablation'],source=str(row['dir']),epochs=len(data)))
  for j,axis in enumerate(axes):axis.set(xlabel='Epoch',ylabel='Training loss' if j==0 else 'Validation loss');axis.set_yscale('log');axis.grid(alpha=.18)
  h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.12),fontsize=6);name='fig_ablation_complete_'+hashlib.sha256('/'.join(key).encode()).hexdigest()[:12];fig.savefig(root/'paper/figures'/f'{name}.pdf');fig.savefig(root/'paper/figures'/f'{name}.png',dpi=180);plt.close(fig)
  title=('/'.join(key)).replace('_','\\_');tex+='\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=.98\\linewidth]{figures/'+name+'.pdf}\n\\caption{\\textbf{Completed fixed-rate ablation trajectories.} '+title+'. Left: training; right: validation. Raw losses for the full method and every currently completed variant, seed 42. No smoothing, test selection or extrapolation. In continual learning the validation set averages seen tasks.}\n\\end{figure}\n'
 target=root/'paper/sections/ablation_curves_live.tex';tmp=target.with_suffix('.tex.partial');tmp.write_text(tex);tmp.replace(target);(root/'results_revision/ablation_curve_audit.json').write_text(json.dumps(audit,indent=2));last=sig;print(json.dumps(dict(curves=len(audit))),flush=True)
 if not a.watch:break
 with (root/'experiments/revision/queue.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_SH);jobs=json.loads((root/'experiments/revision/queue.json').read_text())
 if not any(j['status'] in ['pending','running','waiting_data'] for j in jobs):break
 time.sleep(60)
