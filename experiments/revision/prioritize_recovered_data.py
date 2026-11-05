import fcntl,json
from pathlib import Path
q=Path('revision/queue.json')
with q.with_suffix('.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text())
 for j in jobs:
  if j['status']!='pending' or 'train_image_benchmarks.py' not in ' '.join(j.get('args',[])):continue
  a=j['args'];ds=a[a.index('--dataset')+1];model=a[a.index('--model')+1];method=a[a.index('--method')+1];lr=a[a.index('--lr')+1];base=a[a.index('--base')+1]
  j['priority']=(100 if ds=='imagenet100' else 80 if ds=='core50' else 70 if ds=='domainnet' else 20) if model=='resnet18' and method in ['strata','single'] and lr=='0.0003' and base=='finetune' else 0
 fixed=[j for j in jobs if j['status']!='pending'];pending=sorted([j for j in jobs if j['status']=='pending'],key=lambda j:-j.get('priority',0));q.write_text(json.dumps(fixed+pending,indent=2));print('Next',[(j['id'],j['out']) for j in pending[:4]])
