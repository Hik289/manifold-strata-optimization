"""Append fixed-budget searches; data-dependent activation is handled separately."""
import fcntl,json
from pathlib import Path
q=Path('revision/queue.json');root=Path('..').resolve();jobs_new=[]
def add(ds,model,method,lr,base='finetune'):
 out=f'runs_revision/{ds}__{model}__{base}__{method}__lr{lr}'
 args=['revision/train_image_benchmarks.py','--dataset',ds,'--model',model,'--method',method,'--lr',str(lr),'--base',base,'--epochs','1000','--data','../data','--out',out]
 jobs_new.append(dict(family='continual' if ds in ['domainnet','core50'] else 'vision',out=out,args=args,status='waiting_data',requires=[str(root/'data'/ds/'ready.json')],priority=10 if model=='resnet18' and base=='finetune' and method in ['strata','single'] else 0))
for ds in ['imagenet100','tinyimagenet']:
 for model in ['resnet18','resnet50','vit_s16','convnext_t']:
  for method in ['strata','single','adamw','galore','independent','sgdm','muon']:
   rates=[.03,.1] if method=='sgdm' else [.01,.02] if method=='muon' else [.0003,.001]
   for lr in rates:add(ds,model,method,lr)
for ds in ['domainnet','core50']:
 for base in ['finetune','replay','ewc']:
  for method in ['strata','single','adamw','galore','independent']:
   for lr in [.0003,.001]:add(ds,'resnet18',method,lr,base)
with q.with_suffix('.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j['out'] for j in jobs};nid=max(j['id'] for j in jobs)+1
 for j in jobs_new:
  if j['out'] in outs:continue
  j['id']=nid;nid+=1;jobs.append(j)
 q.write_text(json.dumps(jobs,indent=2));print('Total configurations',len(jobs))
