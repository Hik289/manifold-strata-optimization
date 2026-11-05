import fcntl,json,os,subprocess,sys
from pathlib import Path
variants=['same_rho','reverse_rho','equal_a','no_residual','random_flag','no_transport','frozen_after_warmup','heterogeneous_memory','periodic_svd','least_squares_transport']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3'
for ab in ['heterogeneous_memory','periodic_svd','least_squares_transport']:
 subprocess.run([sys.executable,'revision/train_tabular.py','--dataset','wine_red','--method','strata','--ablation',ab,'--smoke','--epochs','1','--data','../data','--out',f'smoke_breadth/ab_{ab}'],env=env,check=True)
q=Path('revision/queue.json')
with q.with_suffix('.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j['out'] for j in jobs};nid=max(j['id'] for j in jobs)+1
 for ds in ['wine_red','adult']:
  for ab in variants:
   for lr in [.0003,.001]:
    out=f'runs_revision/{ds}__mlp256__strata__{ab}__lr{lr}'
    if out in outs:continue
    args=['revision/train_tabular.py','--dataset',ds,'--method','strata','--ablation',ab,'--lr',str(lr),'--flag-warmup-steps','200','--epochs','1000','--data','../data','--out',out]
    jobs.append(dict(id=nid,family='ablation',out=out,args=args,status='pending',breadth_priority=30 if ds=='wine_red' else 10));nid+=1
 q.write_text(json.dumps(jobs,indent=2));print(json.dumps(dict(total=len(jobs),mechanistic_added=40)))
