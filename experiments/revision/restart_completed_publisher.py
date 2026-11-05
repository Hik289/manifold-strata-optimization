"""Restart only this project's result publisher after an atomic-publication fix."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('..').resolve();stopped=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=p.joinpath('cmdline').read_bytes().split(b'\0');tokens=[v.decode(errors='replace') for v in args if v]
  if '--watch' not in tokens or not any(Path(t).name=='collect_results.py' for t in tokens):continue
  if p.joinpath('cwd').resolve()!=root/'experiments':continue
  os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
 except (OSError,PermissionError):pass
log=(root/'publish_completed.log').open('a');proc=subprocess.Popen([sys.executable,'-u','revision/collect_results.py','--watch'],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);(root/'completed_publisher_pid.json').write_text(json.dumps(dict(stopped=stopped,pid=proc.pid)));print(json.dumps(dict(stopped=stopped,pid=proc.pid)))
