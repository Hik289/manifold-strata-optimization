"""Claim eligible jobs atomically from the shared queue; never touch live jobs."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--families',required=True);p.add_argument('--watch',action='store_true');p.add_argument('--out-prefix',default='');a=p.parse_args()
families=a.families.split(',');q=Path('revision/queue.json').resolve();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',HF_HOME=str(q.parents[2]/'hf_cache'),TOKENIZERS_PARALLELISM='false')
last=None
while True:
 with q.with_suffix('.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());eligible=[j for j in jobs if j['status']=='pending' and j['family'] in families and j['out'].startswith(a.out_prefix)]
  job=min(eligible,key=lambda j:(j['family']==last, -j.get('breadth_priority',0),j['id'])) if eligible else None
  if job:
   job.update(status='running',gpu=a.gpu,worker_pid=os.getpid(),worker_type='breadth',started=time.time());q.write_text(json.dumps(jobs,indent=2));chosen=dict(job);last=job['family']
 if job is None:
  if a.watch:time.sleep(30);continue
  break
 out=Path(chosen['out']);out.mkdir(parents=True,exist_ok=True)
 with (out/'console.log').open('a') as log:
  result=subprocess.run([sys.executable]+chosen['args'],env=env,stdout=log,stderr=subprocess.STDOUT)
 with q.with_suffix('.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());job=next(j for j in jobs if j['id']==chosen['id']);summary=out/'summary.json';good=False
  if result.returncode==0 and summary.exists():
   try:good=json.loads(summary.read_text()).get('complete') is True
   except (OSError,json.JSONDecodeError):pass
  job.update(status='complete' if good else 'failed',returncode=result.returncode,ended=time.time());q.write_text(json.dumps(jobs,indent=2))
 print(json.dumps(dict(id=chosen['id'],status=job['status'])),flush=True)
