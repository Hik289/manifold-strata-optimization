"""Predeclare Single-G rank from tensor storage only, never validation/test.

Reuse the complete STRATA/Single-G/Independent R64 step grids; add six matched
state candidates to test whether extra STRATA storage explains its benefit.
"""
import argparse,fcntl,json,math,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
from train_graph_extra import GNN
from train_stream import optimizer,tensor_state_bytes
p=argparse.ArgumentParser();p.add_argument('--gpu',default='2');a=p.parse_args();root=Path('..').resolve();folder=root/'results_revision/memory_control';folder.mkdir(exist_ok=True);torch.set_num_threads(4)
def measure(method,rank):
 torch.manual_seed(42);model=GNN(500,3,'sage');opt=optimizer(model,SimpleNamespace(method=method,rank=rank,lr=.001,ablation='full'))
 for w in model.parameters():w.grad=torch.zeros_like(w)
 opt.step();return tensor_state_bytes(opt)
target=measure('strata',64);s64=measure('single',64);s65=measure('single',65);slope=s65-s64
shapes=[tuple(w.shape) for w in GNN(500,3,'sage').parameters() if w.ndim==2 and min(w.shape)>=256]
def matrix_bytes(rank):return sum(4*(2*max(shape)*rank+min(shape)*rank+2*min(shape)*min(min(shape),rank+8)) for shape in shapes)
base=s64-matrix_bytes(64);counts={r:base+matrix_bytes(r) for r in range(4,256)};assert counts[65]==s65
rank=min(counts,key=lambda r:abs(counts[r]-target));actual=measure('single',rank);assert counts[rank]==actual and abs(actual-target)/target<.01;assert measure('single',255)==counts[255]
(folder/'calibration.json').write_text(json.dumps(dict(eligible_matrix_shapes=shapes,dense_fallback_constant_bytes=base,counts_by_rank=counts,verified_ranks=[64,65,rank,255],chosen=rank,selection_uses_accuracy=False),indent=2))
# Count is cross-checked against a real final checkpoint before choosing rank.
audit=json.loads((root/'results_revision/checkpoint_resource_audit.json').read_text());old=next(x for x in audit['runs'] if x['out']=='runs_revision/PubMed__sage__strata__lr0.001');assert target==old['optimizer_state_bytes']
plan=dict(seed=42,epochs=1000,dataset='PubMed',model='sage',method='single',rank=rank,rank_selection='closest integer rank in [4,255] by unique optimizer tensor storage including probes; no accuracy inspected',strata_state_bytes=target,single64_state_bytes=s64,single_matched_state_bytes=actual,state_difference_fraction=(actual-target)/target,etas=[.0003,.001],xis=[.001,.1,10.],selection='terminal-100 validation accuracy',reuse='same six-candidate R64 STRATA/Single-G/Independent grids; retain every outcome',reason='rank64 STRATA has extra state versus rank64 Single-G; resource-only capacity control')
manifest=folder/'plan.json'
if manifest.exists():assert json.loads(manifest.read_text())==plan
else:manifest.write_text(json.dumps(plan,indent=2))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4')
for xi in [.001,10.]:
 out=root/f'experiments/smoke_memory_control/single_R{rank}_xi{xi:g}'
 if not (out/'summary.json').exists():subprocess.run([sys.executable,'revision/train_graph_extra.py','--dataset','PubMed','--model','sage','--method','single','--rank',str(rank),'--lr','.001','--flag-xi',str(xi),'--epochs','2','--smoke','--data','../data','--out',str(out)],env=env,check=True,stdout=subprocess.DEVNULL)
 assert json.loads((out/'summary.json').read_text())['complete']
(folder/'smokes.json').write_text(json.dumps(dict(cases=2,rank=rank,all_passed=True)))
q=root/'experiments/revision/queue.json';added=[]
with q.with_suffix('.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j['out'] for j in jobs};nid=max(j['id'] for j in jobs)+1
 for xi in plan['xis']:
  for lr in plan['etas']:
   out=f'runs_revision/PubMed__sage__single__R{rank}__xi{xi:g}__lr{lr}'
   if out in outs:continue
   args=['revision/train_graph_extra.py','--dataset','PubMed','--model','sage','--method','single','--rank',str(rank),'--lr',str(lr),'--flag-xi',str(xi),'--epochs','1000','--data','../data','--out',out]
   jobs.append(dict(id=nid,family='graph',out=out,args=args,status='pending',breadth_priority=130,memory_control=True,protocol_plan=str(manifest)));added.append(nid);nid+=1
 tmp=q.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(q)
if added or not (folder/'activation.json').exists():(folder/'activation.json').write_text(json.dumps(dict(added_ids=added,total=len(jobs),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
print(json.dumps(dict(plan=plan,added=added)))
