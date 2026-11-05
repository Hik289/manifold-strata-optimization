"""Reuse released ablation capacity for the existing nonstationary matrix."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
root=Path('..').resolve();q=root/'experiments/revision/queue.json';pidfile=root/'shift_after_ablation_worker.json'
while True:
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);jobs=json.loads(q.read_text())
 if not any(j['family']=='nonstationary' and j['status']=='pending' for j in jobs):break
 if any(j['family']=='ablation' and j['status'] in ['running','pending'] for j in jobs):time.sleep(30);continue
 free,util=map(float,subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 if free<12000 or util>70:time.sleep(30);continue
 if pidfile.exists():
  try:
   pid=json.loads(pidfile.read_text())['pid'];tokens=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
   if any(t.endswith(b'run_breadth_queue.py') for t in tokens) and b'nonstationary' in tokens and Path(f'/proc/{pid}/cwd').resolve()==root/'experiments':break
  except (OSError,KeyError):pass
 log=(root/'worker_shift_gpu2.log').open('a');child=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu','2','--families','nonstationary'],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);pidfile.write_text(json.dumps(dict(pid=child.pid,trigger='ablation matrix drained; GPU2 free >=12GiB and utilization <=70%',epochs=1000,seed=42),indent=2));print(child.pid,flush=True);break
