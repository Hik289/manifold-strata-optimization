"""Restart only the two figure publishers; preserve all training processes."""
import json,os,signal,subprocess,sys
from pathlib import Path
root=Path('..').resolve();names=['publish_breadth_curves.py','publish_ablation_curves.py'];stopped=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[v.decode(errors='replace') for v in p.joinpath('cmdline').read_bytes().split(b'\0') if v]
  if '--watch' not in args or not any(Path(v).name in names for v in args) or p.joinpath('cwd').resolve()!=root/'experiments':continue
  os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
 except OSError:pass
pids={}
for name in names:
 log=(root/(name.removesuffix('.py')+'.log')).open('a');proc=subprocess.Popen([sys.executable,'-u','revision/'+name,'--watch'],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);pids[name]=proc.pid
print(json.dumps(dict(stopped=stopped,started=pids)))
