"""Add new matched breadth cohorts, preserving all original jobs and budgets."""
import fcntl,json
from pathlib import Path
q=Path('revision/queue.json');new=[]
for ds in ['wine_red','adult','wine_white','covertype']:
 for width in [256,512]:
  for lr in [.0003,.001]:
   for method in ['strata','single','adamw','galore','independent']:
    out=f'runs_revision/{ds}__mlp{width}__{method}__lr{lr}';args=['revision/train_tabular.py','--dataset',ds,'--width',str(width),'--method',method,'--lr',str(lr),'--epochs','1000','--data','../data','--out',out];marker=Path('../data/tabular')/(ds+'.json');new.append(dict(family='tabular',out=out,args=args,status='pending' if marker.exists() else 'waiting_data',requires=[str(marker.resolve())],breadth_priority=20 if width==256 and ds=='wine_red' else 10 if width==256 else 0))
for model in ['lstm','gru','transformer']:
 for lr in [.0003,.001]:
  for method in ['strata','single','adamw','galore','independent']:
   out=f'runs_revision/wikitext2_words__{model}__{method}__lr{lr}';args=['revision/train_sequence.py','--model',model,'--method',method,'--lr',str(lr),'--epochs','1000','--data','../data','--out',out];marker=Path('../data/sequence/wikitext2_words.json');new.append(dict(family='sequence',out=out,args=args,status='pending' if marker.exists() else 'waiting_data',requires=[str(marker.resolve())]))
with q.with_suffix('.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j['out'] for j in jobs};nid=max(j['id'] for j in jobs)+1
 for j in new:
  if j['out'] in outs:continue
  j['id']=nid;nid+=1;jobs.append(j)
 for j in jobs:
  if j['status']=='waiting_data' and j['family'] in ['tabular','sequence'] and all(Path(p).exists() for p in j['requires']):j['status']='pending'
 q.write_text(json.dumps(jobs,indent=2));print(json.dumps(dict(total=len(jobs),new_families=['tabular','sequence'])))
