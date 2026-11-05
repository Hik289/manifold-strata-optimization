import fcntl,json
from pathlib import Path
p=Path('revision/queue.json')
with p.with_suffix('.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);q=json.loads(p.read_text());pending=[j for j in q if j['status']=='pending']
 priority=[j for j in pending if j['family']=='continual' and 'rotated_mnist_cl__finetune__' in j['out'] and ('__single__lr0.001' in j['out'] or '__strata__lr0.001' in j['out'])]
 priority += [j for j in pending if j['family']=='graph' and 'Cora__gcn__' in j['out'] and 'lr0.0003' in j['out'] and any('__'+m+'__' in j['out'] for m in ['strata','single','adamw'])]
 ids={j['id'] for j in priority};fixed=[j for j in q if j['status']!='pending'];rest=[j for j in pending if j['id'] not in ids];p.write_text(json.dumps(fixed+priority+rest,indent=2));print('prioritized',len(priority))
