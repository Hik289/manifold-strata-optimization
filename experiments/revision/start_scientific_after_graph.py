"""Reuse the graph worker's capacity once its full cohort queue drains."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
root=Path('..').resolve();q=root/'experiments/revision/queue.json'
while True:
 with q.with_suffix('.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_SH);jobs=json.loads(q.read_text())
 if not any(j['family']=='graph' and j['status'] in ['pending','running'] for j in jobs):break
 time.sleep(30)
if any(j['family']=='scientific' and j['status']=='pending' for j in jobs):
 log=(root/'worker_scientific_gpu2.log').open('a');proc=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu','2','--families','scientific'],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);(root/'scientific_worker_pid.json').write_text(json.dumps(dict(pid=proc.pid,trigger='graph cohort drained')));print(proc.pid,flush=True)
