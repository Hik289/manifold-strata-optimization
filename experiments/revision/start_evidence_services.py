"""Idempotent CPU postprocessing and deferred training handoff; no notifications."""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--refresh-publisher',action='store_true');a=parser.parse_args()
root=Path('..').resolve();roles={'evidence_publisher':['revision/publish_evidence_results.py','--watch'],'core50_handoff':['revision/start_core50_after_graph.py'],'shift_handoff':['revision/start_shift_after_ablation.py'],'new_family_publisher':['revision/publish_new_families.py','--watch']};out={}
for role,args in roles.items():
 pidfile=root/f'{role}_pid.json';alive=False
 if pidfile.exists():
  try:
   pid=json.loads(pidfile.read_text())['pid'];tokens=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');alive=any(t.endswith(Path(args[0]).name.encode()) for t in tokens) and Path(f'/proc/{pid}/cwd').resolve()==root/'experiments'
  except (OSError,KeyError):pass
 if alive and role in ['evidence_publisher','new_family_publisher'] and a.refresh_publisher:os.kill(pid,signal.SIGTERM);alive=False
 if not alive:
  log=(root/f'{role}.log').open('a');child=subprocess.Popen([sys.executable,'-u',*args],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);pid=child.pid;pidfile.write_text(json.dumps(dict(pid=pid,args=args),indent=2))
 out[role]=pid
print(json.dumps(out))
