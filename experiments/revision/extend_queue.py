"""Append verified graph/Darcy adapters without touching running queue entries."""
import fcntl,json
from pathlib import Path
q=Path('revision/queue.json');lock=q.with_suffix('.lock');new=[]
for ds,model in [('Cora','gcn'),('CiteSeer','gat'),('PubMed','sage'),('Cora','gcnii')]:
 for lr in [.0003,.001]:
  for method in ['strata','single','adamw','galore','independent']:
   out=f'runs_revision/{ds}__{model}__{method}__lr{lr}'
   new.append(dict(family='graph',out=out,args=['revision/train_graph_extra.py','--dataset',ds,'--model',model,'--method',method,'--lr',str(lr),'--data','../data','--out',out],status='pending'))
for lr in [.0003,.001]:
 for method in ['strata','single','adamw','galore','independent']:
  out=f'runs_revision/darcy_generated__fno__{method}__lr{lr}'
  new.append(dict(family='scientific',out=out,args=['revision/train_darcy.py','--method',method,'--lr',str(lr),'--data','../data','--out',out],status='pending'))
with lock.open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j['out'] for j in jobs};nid=max(j['id'] for j in jobs)+1
 for j in new:
  if j['out'] in outs:continue
  j['id']=nid;nid+=1;jobs.append(j)
 q.write_text(json.dumps(jobs,indent=2));print('Total jobs',len(jobs))
