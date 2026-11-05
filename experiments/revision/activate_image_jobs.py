"""Activate queued image jobs only after validation and smoke tests pass.
Exits after all data-waiting jobs are activated; failures are retained in logs.
"""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
q=Path('revision/queue.json');root=Path('..').resolve();verified=set();last={}
while True:
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);jobs=json.loads(q.read_text())
 waiting=[j for j in jobs if j['status']=='waiting_data']
 if not waiting:break
 for ds in ['imagenet100','tinyimagenet','domainnet','core50']:
  marker=root/'data'/ds/'ready.json'
  if ds in verified or not marker.exists():continue
  try:info=json.loads(marker.read_text())
  except (json.JSONDecodeError,OSError):continue
  assert info['ready']
  # Small real-data run checks the complete optimizer and logging path; excluded
  # from formal results. The complete 1000-epoch job is a separate process.
  out=f'smoke_images/{ds}_strata'
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['OMP_NUM_THREADS']='4'
  with (root/f'smoke_images_{ds}.log').open('a') as log:
   rc=subprocess.run([sys.executable,'revision/train_image_benchmarks.py','--dataset',ds,'--model','resnet18','--method','strata','--smoke','--epochs','1','--batch','8','--workers','0','--data','../data','--out',out],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
  if rc!=0:
   print(json.dumps(dict(dataset=ds,smoke_failed=rc)),flush=True);time.sleep(30);continue
  # Model-specific one-step checks are recorded separately; this validates the real-data pipeline.
  verified.add(ds);print(json.dumps(dict(dataset=ds,data_validated=True,smoke_passed=True)),flush=True)
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads(q.read_text());activated=[]
  for j in jobs:
   if j['status']=='waiting_data' and all(Path(p).exists() for p in j['requires']) and any(f'/{ds}/ready.json' in j['requires'][0] for ds in verified):
    j['status']='pending';j['data_validated']=True;activated.append(j['id'])
  high=[j for j in jobs if j['status']=='pending' and j.get('priority',0)>0];ids={j['id'] for j in high};fixed=[j for j in jobs if j['status']!='pending'];rest=[j for j in jobs if j['status']=='pending' and j['id'] not in ids];q.write_text(json.dumps(fixed+high+rest,indent=2))
  if activated:print(json.dumps(dict(activated=activated)),flush=True)
 time.sleep(30)
