"""Reuse idle completed-drift CPU workers; never interrupt a running trial."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('..').resolve();folder=root/'results_revision/new_families';q=root/'experiments/revision/queue.json';changes=[]
for index in range(2):
 path=folder/f'worker_sensor_drift_{index}.json'
 if not path.exists():continue
 old=json.loads(path.read_text());pid=old['pid']
 if old.get('families')=='forecasting,recommendation':continue
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);jobs=json.loads(q.read_text())
 if any(j['status'] in ['pending','running'] and j.get('new_family_extension') and j['family']=='sensor_drift' for j in jobs):break
 if any(j['status']=='running' and j.get('worker_pid')==pid for j in jobs):continue
 load=os.getloadavg()[0];cores=os.cpu_count()
 if load+2>.9*cores:changes.append(dict(index=index,status='waiting CPU capacity',load=load));continue
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');assert b'sensor_drift' in cmd and b'cpu' in cmd;assert Path(f'/proc/{pid}/cwd').resolve()==root/'experiments';os.kill(pid,signal.SIGTERM)
 except OSError:pass
 log=(root/f'worker_new_balanced_{index}.log').open('a');p=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu','cpu','--families','forecasting,recommendation','--out-prefix','runs_revision/new__','--watch'],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'));record=dict(pid=p.pid,family='balanced-after-drift',families='forecasting,recommendation',threads=2,backend='cpu',previous_pid=pid,reason='all60 declared sensor-drift candidates finished; no trial interrupted',time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));path.write_text(json.dumps(record,indent=2));changes.append(record)
(folder/'rebalanced_workers.json').write_text(json.dumps(changes,indent=2));print(json.dumps(changes))
