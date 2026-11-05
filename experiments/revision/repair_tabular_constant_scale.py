"""Before formal trials, give train-constant features unit scale; retain v1."""
import fcntl,hashlib,json,shutil
from pathlib import Path
import numpy as np
r=Path('..').resolve();q=r/'experiments/revision/queue.json'
for ds in ['adult','covertype']:
 with q.with_suffix('.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);jobs=json.loads(q.read_text())
 assert not any(j['status'] in ['running','complete'] and Path(j['out']).name.startswith(ds+'__') for j in jobs),ds
 base=r/'data/tabular';p=base/f'{ds}.npz';m=base/f'{ds}.json';data=dict(np.load(p));mask=data['scale']<1e-5
 if not mask.any():print(json.dumps(dict(dataset=ds,constant_columns=0)));continue
 backup=base/'archive_constant_scale_v1';backup.mkdir(exist_ok=True)
 if (backup/p.name).exists():raise RuntimeError('Repair already has an archive; inspect instead of applying twice')
 shutil.copy2(p,backup/p.name);shutil.copy2(m,backup/m.name);oldscale=data['scale'].copy()
 for s in ['train','validation','test']:data[s+'_x'][:,mask]*=oldscale[mask]
 data['scale'][mask]=1.
 with p.with_suffix('.npz.partial').open('wb') as f:np.savez_compressed(f,**data)
 p.with_suffix('.npz.partial').replace(p);meta=json.loads(m.read_text());meta.update(constant_feature_scale='unit scale on train-constant columns; UNK remains 0/1',preprocessing_revision='constant-scale-v2',sha256=hashlib.sha256(p.read_bytes()).hexdigest());m.write_text(json.dumps(meta,indent=2));print(json.dumps(dict(dataset=ds,constant_columns=int(mask.sum()),sha256=meta['sha256'])))
