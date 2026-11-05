"""Start one full-budget Core50 worker after graph work drains and capacity fits."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
root=Path('..').resolve();q=root/'experiments/revision/queue.json';pidfile=root/'core50_after_graph_worker.json'
while True:
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);jobs=json.loads(q.read_text())
 if not any(j['out'].startswith('runs_revision/core50__') and j['status']=='pending' for j in jobs):break
 graph_busy=any(j['family']=='graph' and j['status'] in ['pending','running'] for j in jobs)
 if graph_busy:time.sleep(30);continue
 try:
  query=subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');free,util=map(float,query)
 except (ValueError,subprocess.SubprocessError):time.sleep(30);continue
 if free<12000 or util>70:time.sleep(30);continue
 if not (root/'data/core50/ready.json').exists():raise RuntimeError('Core50 manifest not ready')
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads(q.read_text())
  for j in jobs:
   if j['status']=='pending' and j['out'].startswith('runs_revision/core50__') and '__finetune__' in j['out'] and '__lr0.0003' in j['out'] and any('__'+m+'__' in j['out'] for m in ['strata','single']):j['breadth_priority']=100
  temp=q.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(jobs,indent=2));temp.replace(q)
 if pidfile.exists():
  try:
   pid=json.loads(pidfile.read_text())['pid'];tokens=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
   if any(t.endswith(b'run_breadth_queue.py') for t in tokens) and b'runs_revision/core50__' in tokens:break
  except (OSError,KeyError):pass
 log=(root/'worker_core50_gpu2.log').open('a');child=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu','2','--families','continual','--out-prefix','runs_revision/core50__'],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);pidfile.write_text(json.dumps(dict(pid=child.pid,trigger='graph drained; GPU2 free memory >=12GiB, utilization <=70%',seed=42,budget_epochs=1000)));print(child.pid,flush=True);break
