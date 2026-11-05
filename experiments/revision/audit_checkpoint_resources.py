"""Measure archived optimizer tensor storage, including probes, without CUDA."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--root',default='..');p.add_argument('--snapshot');p.add_argument('--out');a=p.parse_args();root=Path(a.root).resolve()
queue=Path(a.snapshot) if a.snapshot else root/'snapshot_light/queue_snapshot.json';jobs=json.loads(queue.read_text());result=[]
def tensor_bytes(value):
 seen=set()
 def visit(v):
  if torch.is_tensor(v):
   storage=v.untyped_storage();key=(str(v.device),storage.data_ptr(),storage.nbytes())
   if key in seen:return 0
   seen.add(key);return storage.nbytes()
  if isinstance(v,dict):return sum(visit(x) for x in v.values())
  if isinstance(v,(list,tuple)):return sum(visit(x) for x in v)
  return 0
 return visit(value)
for job in jobs:
 if job['status']!='complete':continue
 run=root/'experiments'/job['out'];config=json.loads((run/'config.json').read_text());rows=[json.loads(x) for x in (run/'log.jsonl').read_text().splitlines()];budget=int(config.get('epochs',config.get('units',len(rows))))
 candidates=[run/'checkpoints'/f'epoch_{budget:04d}.pt',run/'checkpoints'/f'unit_{budget:04d}.pt',run/'checkpoints/latest.pt'];checkpoint=next((f for f in candidates if f.exists()),None)
 entry=dict(id=job['id'],out=job['out'],family=job['family'],budget=budget,observed_units=len(rows),peak_cuda_allocated_bytes=max(r['peak_bytes'] for r in rows),mean_epoch_seconds=sum(r['wall_seconds'] for r in rows)/len(rows),timing_scope='shared GPU; recorded epoch body, not isolated throughput')
 if checkpoint:
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);age=ck.get('epoch',ck.get('unit'));assert age==budget,(checkpoint,age,budget)
  entry.update(checkpoint=str(checkpoint),checkpoint_size=checkpoint.stat().st_size,optimizer_state_bytes=tensor_bytes(ck['optimizer']['state']),model_state_bytes=tensor_bytes(ck['model']),source='final checkpoint unique tensor storages; probes included')
  del ck
 else:entry.update(source='unavailable',optimizer_state_bytes=None)
 result.append(entry)
out=Path(a.out) if a.out else root/'results_revision/checkpoint_resource_audit.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest(),runs=result),indent=2));print(json.dumps(dict(audited=len(result),checkpoint_measured=sum(x['source']!='unavailable' for x in result),out=str(out))))
