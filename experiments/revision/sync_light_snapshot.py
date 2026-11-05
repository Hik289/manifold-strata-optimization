"""Copy logs/configs/summaries only; original checkpoints remain on the server."""
import argparse,fcntl,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);dst=root/'snapshot_light';dst.mkdir(exist_ok=True)
with (root/'experiments/revision/queue.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_SH);jobs=json.loads((root/'experiments/revision/queue.json').read_text())
(dst/'queue_snapshot.json').write_text(json.dumps(jobs,indent=2))
for j in jobs:
 if j['status'] not in ['running','complete','failed']:continue
 src=root/'experiments'/j['out'];target=dst/j['out'];target.mkdir(parents=True,exist_ok=True)
 for name in ['config.json','summary.json','log.jsonl']:
  f=src/name
  if f.exists():shutil.copyfile(f,target/name)
print(json.dumps(dict(copied=sum(j['status'] in ['running','complete','failed'] for j in jobs))))
