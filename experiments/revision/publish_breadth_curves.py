"""Publish only completed five-method breadth cohorts; optional durable watcher."""
import argparse,collections,csv,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',default='..');p.add_argument('--snapshot');p.add_argument('--watch',action='store_true');a=p.parse_args();root=Path(a.root).resolve();last=None
colors={'strata':'#D55E00','single':'#0072B2','adamw':'#333333','galore':'#009E73','independent':'#CC79A7'};names={'strata':'STRATA','single':'Single-G','adamw':'AdamW','galore':'GaLore','independent':'Independent'};methods=list(colors)
plt.rcParams.update({'font.size':7,'axes.titlesize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
while True:
 ledger=root/'results_revision/completed_matrix.csv'
 if not ledger.exists():
  if a.watch:time.sleep(30);continue
  raise FileNotFoundError(ledger)
 signature=ledger.stat().st_mtime_ns
 if a.watch and signature==last:time.sleep(60);continue
 rows=list(csv.DictReader(ledger.open()));groups=collections.defaultdict(dict)
 for row in rows:
  if row['family'] not in ['graph','tabular','sequence','language','continual'] or row['ablation']!='full' or row['method'] not in methods:continue
  source=Path(row['source'])
  if a.snapshot:source=Path(a.snapshot)/'runs_revision'/source.name
  if not (source/'config.json').exists():continue
  key=(row['task'],row['model'],row['base'],row['rank']);row['dir']=source;row['lr']=float(row['lr']);groups[key][(row['method'],row['lr'])]=row
 cohorts=[]
 for key,items in sorted(groups.items()):
  if all((m,lr) in items for m in methods for lr in [.0003,.001]):
   chosen={m:(max if items[(m,.0003)]['metric']=='acc' else min)([items[(m,lr)] for lr in [.0003,.001]],key=lambda r:float(r['val'])) for m in methods};protocol='two-rate validation selection'
  elif all((m,.0003) in items for m in methods):chosen={m:items[(m,.0003)] for m in methods};protocol='fixed rate 0.0003; search incomplete'
  else:continue
  cohorts.append((key,chosen,protocol))
 figdir=root/'paper/figures';figdir.mkdir(exist_ok=True);audit=[];tex='\\subsection{New completed breadth-cohort loss curves}\n\\label{app:breadth-loss-updates}\n'
 tex+='Every displayed cohort contains five completed full-budget methods. Learning rates are selected on validation only after both predeclared rates have completed for every method; otherwise a completed common-rate cohort is labeled fixed-rate. Each curve is traced to its complete log in the accompanying audit.\n'
 if not cohorts:tex+='No complete five-method breadth cohort is available at this snapshot.\n'
 for offset in range(0,len(cohorts),2):
  chunk=cohorts[offset:offset+2];fig,axs=plt.subplots(len(chunk),2,figsize=(6,2.8*len(chunk)),squeeze=False,layout='constrained');captions=[]
  for i,(key,chosen,protocol) in enumerate(chunk):
   task,model,base,rank=key;setting=f'{task}/{model}'+(f'/{base}' if base else '');captions.append(f'{setting}: {protocol}')
   for method,row in chosen.items():
    logfile=row['dir']/'log.jsonl';data=[json.loads(s) for s in logfile.read_text().splitlines()];xkey='tokens' if 'tokens' in data[-1] else 'epoch';x=[r[xkey] for r in data];train=[r['train_loss'] for r in data];val=[r.get('val_loss',np.mean([z['loss'] for z in r.get('val_per_task',[])]) if r.get('val_per_task') else float('nan')) for r in data]
    axs[i,0].plot(x,train,label=names[method],color=colors[method],lw=.85);axs[i,1].plot(x,val,label=names[method],color=colors[method],lw=.85);audit.append(dict(task=task,model=model,base=base,rank=rank,method=method,lr=row['lr'],protocol=protocol,source=str(logfile),sha256=hashlib.sha256(logfile.read_bytes()).hexdigest(),points=len(data)))
   axs[i,0].set_title((f'Rotated-MNIST / '+{'finetune':'fine-tuning','replay':'replay'}.get(base,base)) if task=='rotated_mnist_cl' else f'{task} / {model} / R{rank}')
   for j in [0,1]:axs[i,j].set_xlabel('Training tokens' if xkey=='tokens' else 'Epoch');axs[i,j].grid(alpha=.18);axs[i,j].set_yscale('log')
   axs[i,0].set_ylabel('Training loss');axs[i,1].set_ylabel('Validation loss')
  hs,ls=axs[0,0].get_legend_handles_labels();fig.legend(hs,ls,ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.06));name=f'fig_breadth_complete_{offset//2:02d}';fig.savefig(figdir/(name+'.pdf'));fig.savefig(figdir/(name+'.png'),dpi=180);plt.close(fig)
  text='; '.join(captions).replace('_','\\_')
  continual_note=' Continual validation averages seen tasks, so its evaluation distribution changes at task boundaries.' if any(key[2] for key,_,_ in chunk) else ''
  tex+='\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=.98\\linewidth]{figures/'+name+'.pdf}\n\\caption{\\textbf{Completed breadth-cohort training and validation losses.} '+text+'. Left: training; right: validation. Raw complete trajectories, seed 42.'+continual_note+' No smoothing or extrapolation.}\n\\end{figure}\n'
 dst=root/'paper/sections/breadth_curves_live.tex';tmp=dst.with_suffix('.tex.partial');tmp.write_text(tex);tmp.replace(dst);(root/'results_revision/breadth_curve_audit.json').write_text(json.dumps(audit,indent=2));last=signature;print(json.dumps(dict(completed_cohorts=len(cohorts),curves=len(audit))),flush=True)
 if not a.watch:break
 with (root/'experiments/revision/queue.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_SH);q=json.loads((root/'experiments/revision/queue.json').read_text())
 if not any(j['status'] in ['pending','running','waiting_data'] for j in q) and sum(j['status']=='complete' for j in q)<=len(rows):break
 time.sleep(60)
