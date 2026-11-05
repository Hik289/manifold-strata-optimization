"""Recompute means from raw trajectories; choose configs without test metrics."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'analysis/revision_20261101';out.mkdir(exist_ok=True)
pinn=[]
for task in ['burgers','burgersg2','poisson']:
 for method in ['strata','single_grassmann']:
  cells=[]
  for f in (root/'experiments/runs2_grid_eps1e6').glob(f'{task}__{method}__*/log.jsonl'):
   rows=[json.loads(s) for s in f.read_text().splitlines()];tail=rows[-100:];pde=sum(r.get('loss_pde',float('nan')) for r in tail)/len(tail);rel=sum(r.get('rel_l2',float('nan')) for r in tail)/len(tail)
   if math.isfinite(pde):cells.append((pde,rel,f.parent.name,len(rows)))
  pinn.append(dict(task=task,method=method,cells=sorted(cells)))
(out/'pinn_selection_audit.json').write_text(json.dumps(pinn,indent=2))
cl=[]
for folder in ['runs3_grid','runs3_cnn']:
 for f in (root/'experiments'/folder).glob('*_log.jsonl'):
  name=f.name[:-10];parts=name.split('__')
  if len(parts)<3:continue
  rows=[json.loads(s) for s in f.read_text().splitlines()];tail=rows[-100:];means={k:sum(r[k] for r in tail)/len(tail) for k in ['val_acc','test_acc','train_loss'] if all(k in r for r in tail)}
  cl.append(dict(task=parts[0],method=parts[1],cell=name,epochs=len(rows),**means))
(out/'classification_selection_audit.json').write_text(json.dumps(cl,indent=2))
print('Recomputed',sum(len(x['cells']) for x in pinn),'PINN cells and',len(cl),'classifier cells')
