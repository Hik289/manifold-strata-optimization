import collections,fcntl,json,os,time
from pathlib import Path
r=Path(str(Path(__file__).resolve().parents[2]));q=r/'experiments/revision/queue.json'
with q.with_suffix('.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_SH);jobs=json.loads(q.read_text())
out=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),counts=dict(collections.Counter(j['status'] for j in jobs)),running=[],datasets={},failed=[])
for j in jobs:
 if j['status'] in ['running','failed']:
  d=r/'experiments'/j['out'];p=d/'log.jsonl';last={}
  if p.exists():
   lines=p.read_text().splitlines()
   for line in reversed(lines[-2:]):
    try:last=json.loads(line);break
    except json.JSONDecodeError:pass
  row=dict(id=j['id'],run=j['out'],epoch=last.get('epoch'),unit=last.get('unit'),val_acc=last.get('val_acc'),test_acc=last.get('test_acc'),val_ppl=last.get('val_ppl'),log_age_seconds=round(time.time()-p.stat().st_mtime) if p.exists() else None,worker_pid=j.get('worker_pid'))
  try:row['worker_alive']=any(n in Path(f'/proc/{j["worker_pid"]}/cmdline').read_text() for n in ['run_queue.py','run_breadth_queue.py'])
  except (KeyError,OSError):row['worker_alive']=False
  if j['status']=='failed':
   c=d/'console.log';row['error_tail']=c.read_text(errors='replace').splitlines()[-5:] if c.exists() else [];out['failed'].append(row)
  else:out['running'].append(row)
for name in ['imagenet100','tinyimagenet','domainnet','core50']:
 d=r/'data'/name;ready=d/'ready.json';info=dict(ready=ready.exists())
 if ready.exists():
  try:meta=json.loads(ready.read_text());info.update(counts=meta['counts'],n_classes=meta['n_classes'])
  except json.JSONDecodeError:pass
 info['archives']={p.name:p.stat().st_size for p in (d/'archives').glob('*') if p.is_file()}
 log=r/f'prepare_{name}.log'
 if log.exists():info['last_log']=log.read_text(errors='replace').splitlines()[-3:]
 out['datasets'][name]=info
print(json.dumps(out,indent=2))
(r/'results_revision/progress_snapshot.json').write_text(json.dumps(out,indent=2))
