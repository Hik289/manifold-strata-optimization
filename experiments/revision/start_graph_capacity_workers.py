"""Start at most one extra graph worker per low-utilization GPU, idempotently.

Read-only capacity checks precede starts. Existing jobs are never interrupted.
Each worker uses the locked authoritative queue and the same full budgets.
"""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path('..').resolve();report=root/'results_revision/evidence_extension';out={};samples=[]
for _ in range(3):
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 samples.append({int(x.split(',')[0]):list(map(float,x.split(',')[1:])) for x in text.splitlines()});time.sleep(1)
cores=os.cpu_count();load=os.getloadavg()[0]
for gpu in [2,3]:
 pidfile=report/f'capacity_worker_gpu{gpu}.json';alive=False
 if pidfile.exists():
  try:
   pid=json.loads(pidfile.read_text())['pid'];tokens=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');alive=any(t.endswith(b'run_breadth_queue.py') for t in tokens) and b'graph' in tokens and Path(f'/proc/{pid}/cwd').resolve()==root/'experiments'
  except (OSError,KeyError):pass
 if alive:out[gpu]=dict(status='already running',pid=pid);continue
 capacity=all(s[gpu][0]>=16000 and s[gpu][1]<=55 for s in samples)
 if not capacity or load+8>.9*cores:
  out[gpu]=dict(status='not started: capacity threshold');continue
 log=(root/f'worker_graph_capacity_gpu{gpu}.log').open('a');child=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu',str(gpu),'--families','graph','--watch'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);record=dict(pid=child.pid,gpu=gpu,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),samples=samples,load=load,cores=cores);pidfile.write_text(json.dumps(record,indent=2));out[gpu]=dict(status='started',pid=child.pid);load+=4
print(json.dumps(out))
