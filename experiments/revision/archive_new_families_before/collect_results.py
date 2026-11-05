"""Publish completed runs only into a full-matrix paper appendix and CSV.
Use --watch for a bounded-to-queue-lifetime postprocessor. Partial trajectories
are counted in status, never converted into endpoint results.
"""
import argparse,csv,fcntl,json,math,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='..');p.add_argument('--watch',action='store_true');a=p.parse_args();root=Path(a.root).resolve();last=None
while True:
 with (root/'experiments/revision/queue.lock').open('a+') as queue_lock:
  fcntl.flock(queue_lock,fcntl.LOCK_SH)
  jobs=json.loads((root/'experiments/revision/queue.json').read_text())
 signature=tuple((j['id'],j['status']) for j in jobs)
 if a.watch and signature==last and any(j['status'] in ['running','pending','waiting_data'] for j in jobs):
  time.sleep(30);continue
 complete=[]
 for j in jobs:
  f=root/'experiments'/j['out']/'summary.json'
  if j['status']!='complete' or not f.exists():continue
  summary=json.loads(f.read_text())
  if not summary.get('complete'):continue
  run=f.parent;cfg=json.loads((run/'config.json').read_text());rows=[json.loads(s) for s in (run/'log.jsonl').read_text().splitlines()];tail=rows[-100:]
  if 'val_acc' in rows[-1]:metric='acc';val=sum(r['val_acc'] for r in tail)/len(tail);test=sum(r['test_acc'] for r in tail)/len(tail)
  elif 'val_loss' in rows[-1]:metric='PPL';vv=sum(r['val_loss'] for r in tail)/len(tail);tt=sum(r['test_loss'] for r in tail)/len(tail);val=math.exp(min(vv,50)) if math.isfinite(vv) else vv;test=math.exp(min(tt,50)) if math.isfinite(tt) else tt
  else:metric='relL2';val=sum(r['val']['rel_l2'] for r in tail)/len(tail);test=sum(r['test']['rel_l2'] for r in tail)/len(tail)
  complete.append(dict(id=j['id'],family=j['family'],task=cfg.get('task',cfg.get('dataset','Darcy-generated')),model=cfg.get('model',cfg.get('architecture','FNO')),method=cfg['method'],base=cfg.get('base',''),ablation=cfg.get('ablation','full'),rank=cfg.get('rank',64),lr=cfg['lr'],units=len(rows),metric=metric,val=val,test=test,mean_seconds=sum(r['wall_seconds'] for r in rows)/len(rows),peak_bytes=max(r['peak_bytes'] for r in rows),source=str(run)))
 signature=tuple((j['id'],j['status']) for j in jobs)
 if signature!=last:
  dst=root/'paper/sections';dst.mkdir(parents=True,exist_ok=True);report=root/'results_revision';report.mkdir(exist_ok=True)
  if complete:
   csv_tmp=report/f'completed_matrix.{os.getpid()}.csv.tmp'
   with csv_tmp.open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(complete[0]));w.writeheader();w.writerows(complete)
   csv_tmp.replace(report/'completed_matrix.csv')
  counts={s:sum(j['status']==s for j in jobs) for s in ['pending','waiting_data','running','complete','failed']};status_tmp=report/f'status.{os.getpid()}.json.tmp';status_tmp.write_text(json.dumps(dict(counts=counts,completed=complete),indent=2));status_tmp.replace(report/'status.json')
  tex='\\section{New Long-Horizon Runs: Completion Ledger}\n\\label{app:new-completions}\nOnly runs that completed their prescribed budget are listed. All configurations use seed 42; no statistical significance is inferred. Endpoints are means over the terminal 100 epochs or evaluation units. These are individual configurations, not test-selected optima. GPT-2 language units are predefined ten-step units, not corpus epochs; the separate WikiText-2 sequence runs use complete corpus epochs.\n'
  if not complete:tex+='No new formal run has completed its full budget at this snapshot. Partial trajectories and smoke tests are excluded.\n'
  else:
   tex+='\\begin{longtable}{rlllrrrr}\n\\caption{Completed configurations. Full model, rank, base method, ablation, runtime and source paths are retained in the accompanying completion matrix. Accuracy is higher-is-better; PPL and relative L2 are lower-is-better.}\\\\\n\\toprule ID & Task & Method & Metric & lr & Units & Val & Test \\\\ \\midrule\\endfirsthead\n\\toprule ID & Task & Method & Metric & lr & Units & Val & Test \\\\ \\midrule\\endhead\n'
   for r in complete:
    task=r['task'].replace('_','\\_');method=r['method'].replace('_','\\_');tex+=f"{r['id']} & {task} & {method} & {r['metric']} & {r['lr']:.0e} & {r['units']} & {r['val']:.5f} & {r['test']:.5f} \\\\\n"
   tex+='\\bottomrule\\end{longtable}\n'
  tmp=dst/f'new_results.{os.getpid()}.tex.tmp';tmp.write_text(tex);tmp.replace(dst/'new_results.tex');last=signature
  print(json.dumps(counts),flush=True)
 if not a.watch or not any(j['status'] in ['running','pending','waiting_data'] for j in jobs):break
 time.sleep(30)
