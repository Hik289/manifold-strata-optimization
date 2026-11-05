"""Freeze and launch balanced follow-up experiments motivated by validation only."""
import argparse,ast,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch

p=argparse.ArgumentParser();p.add_argument('--gpu',default='2');a=p.parse_args()
root=Path('..').resolve();q=root/'experiments/revision/queue.json';folder=root/'results_revision/evidence_extension';folder.mkdir(parents=True,exist_ok=True)
variants=['same_rho','reverse_rho','equal_a','no_residual','random_flag','no_transport','frozen_after_warmup','heterogeneous_memory','periodic_svd','least_squares_transport']
methods=['strata','single','adamw','galore','independent'];rates=[.0003,.001]
plan=dict(seed=42,epochs=1000,selection='terminal-100 validation accuracy; retain all test results',
 motivation='PubMed/SAGE has terminal validation STRATA .71792 versus Single-G .70666. White-Wine has positive validation but negative paired test, so neither is a universal winner.',
 graph_matrix=dict(datasets=['Cora','CiteSeer','PubMed'],models=['gcn','gat','sage','gcnii'],methods=methods,rates=rates),
 ablation=dict(dataset='PubMed',model='sage',variants=variants,rates=rates),
 xi_sweep=dict(dataset='PubMed',model='sage',methods=['strata','single','independent'],additional_xi=[.1,10.],default_xi=.001,rates=rates),
 caveats=['The xi sweep is exploratory, not a retroactive change to defaults.','A fixed random flag still has variance resets; freeze also stops resets.','No multi-seed or significance claim.','The full matrix is retained regardless of sign.'])
manifest=folder/'plan.json'
if manifest.exists():assert json.loads(manifest.read_text())==plan
else:manifest.write_text(json.dumps(plan,indent=2))

# Verify that the optional xi control preserves all five original default paths.
import train_stream as revised
oldpath=root/'experiments/revision/archive_evidence_before/train_stream.py'
oldast=ast.parse(oldpath.read_text());node=next(x for x in oldast.body if isinstance(x,ast.FunctionDef) and x.name=='optimizer')
namespace=dict(revised.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),str(oldpath),'exec'),namespace);old_optimizer=namespace['optimizer']
torch.set_num_threads(4);equiv=[]
for method in methods:
 torch.manual_seed(42);left=torch.nn.Linear(96,128);right=torch.nn.Linear(96,128);right.load_state_dict(left.state_dict());args=SimpleNamespace(method=method,lr=.0003,rank=16,ablation='full')
 old=old_optimizer(left,args);new=revised.optimizer(right,args)
 for step in range(55):
  for x,y in zip(left.parameters(),right.parameters()):
   grad=torch.randn_like(x);x.grad=grad;y.grad=grad.clone()
  old.step();new.step()
  assert all(torch.equal(x,y) for x,y in zip(left.parameters(),right.parameters())),(method,step)
 equiv.append(dict(method=method,steps=55,parameter_max_error=0.))
(folder/'default_equivalence.json').write_text(json.dumps(equiv,indent=2))

env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4')
smokes=[]
for kind in ['gcn','sage','gcnii','gat']:
 smokes.append(['--dataset','PubMed' if kind=='gat' else 'Cora','--model',kind,'--method','strata'])
for ab in variants:
 smokes.append(['--dataset','PubMed','--model','sage','--method','strata','--ablation',ab,'--flag-warmup-steps','1'])
for xi in [.1,10.]:
 for method in ['strata','single','independent']:
  smokes.append(['--dataset','PubMed','--model','sage','--method',method,'--flag-xi',str(xi)])
code_hash=hashlib.sha256((root/'experiments/revision/train_graph_extra.py').read_bytes()).hexdigest()[:10]
for i,extra in enumerate(smokes):
 out=root/f'experiments/smoke_evidence_{code_hash}/case_{i:02d}'
 if not (out/'summary.json').exists():
  subprocess.run([sys.executable,'revision/train_graph_extra.py',*extra,'--lr','.0003','--epochs','2','--smoke','--data','../data','--out',str(out)],env=env,check=True)
 assert json.loads((out/'summary.json').read_text()).get('complete')
(folder/'smokes.json').write_text(json.dumps(dict(cases=len(smokes),source_hash=code_hash,all_passed=True),indent=2))

def entry(ds,kind,method,lr,ab='full',xi=None):
 suffix='' if ab=='full' and xi is None else f'__{ab}' if xi is None else f'__xi{xi:g}'
 out=f'runs_revision/{ds}__{kind}__{method}{suffix}__lr{lr}'
 args=['revision/train_graph_extra.py','--dataset',ds,'--model',kind,'--method',method,'--lr',str(lr),'--epochs','1000','--rank','64','--data','../data','--out',out]
 if ab!='full':args+=['--ablation',ab,'--flag-warmup-steps','200']
 if xi is not None:args+=['--flag-xi',str(xi)]
 priority=120 if ab!='full' else 110 if xi is not None else 100 if ds=='PubMed' else 90
 return dict(family='graph',out=out,args=args,status='pending',breadth_priority=priority,evidence_extension=True,protocol_plan=str(manifest))
new=[]
for ds in plan['graph_matrix']['datasets']:
 for kind in plan['graph_matrix']['models']:
  for lr in rates:
   for method in methods:new.append(entry(ds,kind,method,lr))
for ab in variants:
 for lr in rates:new.append(entry('PubMed','sage','strata',lr,ab=ab))
for xi in [.1,10.]:
 for lr in rates:
  for method in ['strata','single','independent']:new.append(entry('PubMed','sage',method,lr,xi=xi))
with q.with_suffix('.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j['out'] for j in jobs};nid=max(j['id'] for j in jobs)+1;added=[]
 for job in new:
  if job['out'] in outs:continue
  job['id']=nid;nid+=1;jobs.append(job);outs.add(job['out']);added.append(job['id'])
 # Close the second-width validation follow-up before launching more long Covertype runs.
 for job in jobs:
  if job['status']=='pending' and job['family']=='tabular' and 'wine_white__mlp512__' in job['out']:job['breadth_priority']=40
 temp=q.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(jobs,indent=2));temp.replace(q)
(folder/'activation.json').write_text(json.dumps(dict(added_ids=added,total=len(jobs),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))

pidfile=folder/'additional_worker.json';alive=False
if pidfile.exists():
 try:
  pid=json.loads(pidfile.read_text())['pid'];tokens=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
  alive=any(t.endswith(b'run_breadth_queue.py') for t in tokens) and Path(f'/proc/{pid}/cwd').resolve()==root/'experiments'
 except (OSError,KeyError):pass
if not alive:
 logfile=(root/'worker_graph_evidence_gpu2.log').open('a')
 child=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu',a.gpu,'--families','graph','--watch'],stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 pidfile.write_text(json.dumps(dict(pid=child.pid,gpu=a.gpu),indent=2))
print(json.dumps(dict(new_jobs=len(added),matrix_total=len(jobs),smokes=len(smokes),default_equivalence=equiv)),flush=True)
