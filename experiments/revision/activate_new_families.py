"""Freeze six new real-data settings, validate every method, then enqueue.
Full outcomes, failed candidates and negative comparisons remain in the record.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--no-workers',action='store_true');a=p.parse_args();root=Path('..').resolve();folder=root/'results_revision/new_families';folder.mkdir(exist_ok=True);methods=['strata','single','adamw','galore','independent'];settings=[('forecasting','ETTh1','static'),('forecasting','ETTh2','static'),('recommendation','ml100k','static'),('recommendation','ml1m','static'),('sensor_drift','gas_drift','finetune'),('sensor_drift','gas_drift','replay')]
geo=[dict(lr=lr,xi=xi) for xi in [.001,.1,10.] for lr in [.0003,.001]];dense=[dict(lr=lr,xi=None) for lr in [.0001,.0003,.001,.003,.01,.03]];grids={m:geo if m in ['strata','single','independent'] else dense for m in methods};sources={x:json.loads((root/'data/new_families'/f'{x}.json').read_text()) for x in ['ETTh1','ETTh2','ml100k','ml1m','gas_drift']}
plan=dict(version=1,seed=42,epochs=1000,backend='CPU float32; two intra-op threads per run',rank=32,methods=methods,settings=[dict(family=f,task=t,base=b) for f,t,b in settings],candidate_grids=grids,candidates_per_method=6,total_configurations=180,selection='terminal100 mean primary validation metric; no test-based configuration selection',failure_policy='all complete or failed candidates exhaust the declared search; failures have no fabricated endpoint; a method with no complete candidate has no selected score',primary_metrics=dict(forecasting='normalized MSE lower is better',recommendation='raw rating RMSE lower is better',sensor_drift='equal-domain mean seen-task accuracy higher is better'),hypotheses=dict(forecasting='temporally correlated forecasting gradients may contain persistent and faster directions',recommendation='low-dimensional user/item factors may permit a useful state-quality tradeoff',sensor_drift='real sensor drift with and without replay tests adaptation and retention'),reporting='show every setting in the complete appendix and ledger; focus main-text case studies with explicit scope, without deleting unfavorable outcomes; no guaranteed family winner and no statistical-significance claim',data_sha256={k:v['array_sha256'] for k,v in sources.items()},data_sources={k:v['source_url'] for k,v in sources.items()},controlled_budgets='six distinct candidates for each method; geometric methods search two learning rates x three flag steps; AdamW/GaLore search six learning rates; this equal-search-count design is disclosed rather than described as an identical hyperparameter space')
manifest=folder/'plan.json'
if manifest.exists():assert json.loads(manifest.read_text())==plan
else:manifest.write_text(json.dumps(plan,indent=2))
trainer=root/'experiments/revision/train_new_families.py';sourcehash=hashlib.sha256(trainer.read_bytes()).hexdigest();smokes=[]
for family,task,base in settings:
 for method in methods:smokes.append((family,task,base,method,.1 if method in ['strata','single','independent'] else None))
for family,task,base in [settings[0],settings[2],settings[4]]:
 for method in ['strata','single','independent']:smokes.append((family,task,base,method,10.))
def smoke(case):
 family,task,base,method,xi=case;name=f'{task}_{base}_{method}_xi{xi}';out=root/'experiments'/f'smoke_new_families_{sourcehash[:10]}'/name;cmd=[sys.executable,'revision/train_new_families.py','--task',task,'--base',base,'--method',method,'--lr','.001','--rank','32','--epochs','2','--smoke','--device','cpu','--threads','2','--data','../data','--out',str(out)]
 if xi is not None:cmd+=['--flag-xi',str(xi)]
 out.mkdir(parents=True,exist_ok=True)
 if not (out/'summary.json').exists():
  with (out/'console.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
 s=json.loads((out/'summary.json').read_text());assert s['complete'] and s['smoke'];return dict(task=task,base=base,method=method,xi=xi,out=str(out),epochs=s['final']['epoch'],metric=s['final']['metric'],optimizer_state_bytes=s['final']['optimizer_state_bytes'])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:checks=list(pool.map(smoke,smokes))
(folder/'smokes.json').write_text(json.dumps(dict(source_sha256=sourcehash,cases=checks,all_passed=True),indent=2))
q=root/'experiments/revision/queue.json';added=[]
with q.with_suffix('.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j['out'] for j in jobs};nid=max(j['id'] for j in jobs)+1
 for family,task,base in settings:
  for i in range(6):
   for method in methods:
    c=grids[method][i];xi=c['xi'];lr=c['lr'];out=f'runs_revision/new__{task}__{base}__{method}__R32__xi{xi}__lr{lr}'
    if out in outs:continue
    args=['revision/train_new_families.py','--task',task,'--base',base,'--method',method,'--lr',str(lr),'--rank','32','--epochs','1000','--device','cpu','--threads','2','--batch','2048' if family=='recommendation' else '512','--data','../data','--out',out]
    if xi is not None:args+=['--flag-xi',str(xi)]
    jobs.append(dict(id=nid,family=family,out=out,args=args,status='pending',new_family_extension=True,protocol_plan=str(manifest),breadth_priority=50));outs.add(out);added.append(nid);nid+=1
 tmp=q.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(q)
if added or not (folder/'activation.json').exists():(folder/'activation.json').write_text(json.dumps(dict(added_ids=added,total=len(jobs),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
workers={};free_kib=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')));load=os.getloadavg()[0];cores=os.cpu_count();planned=0
if not a.no_workers:
 for family in ['sensor_drift','forecasting','recommendation']:
  for index in range(2):
   key=f'{family}_{index}';pidfile=folder/f'worker_{key}.json';alive=False;worker_family=family
   if pidfile.exists():
    try:
     worker_info=json.loads(pidfile.read_text());pid=worker_info['pid'];worker_family=worker_info.get('families',family);tokens=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');alive=any(t.endswith(b'run_breadth_queue.py') for t in tokens) and worker_family.encode() in tokens and b'cpu' in tokens and Path(f'/proc/{pid}/cwd').resolve()==root/'experiments'
    except (OSError,KeyError):pass
   if alive:workers[key]=dict(pid=pid,status='already running');continue
   if load+2*(planned+1)>.85*cores or free_kib<8*1024**2:workers[key]=dict(status='waiting CPU capacity',load=load,cores=cores,available_kib=free_kib);continue
   log=(root/f'worker_new_{key}.log').open('a');child=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu','cpu','--families',worker_family,'--out-prefix','runs_revision/new__','--watch'],cwd=root/'experiments',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'));record=dict(pid=child.pid,family=family,families=worker_family,threads=2,backend='cpu',start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));pidfile.write_text(json.dumps(record,indent=2));workers[key]=dict(pid=child.pid,status='started');planned+=1
(folder/'worker_activation.json').write_text(json.dumps(dict(workers=workers,initial_load=load,cores=cores,available_kib=free_kib),indent=2));print(json.dumps(dict(added=len(added),queue_total=len(jobs),smokes=len(checks),workers=workers)),flush=True)
