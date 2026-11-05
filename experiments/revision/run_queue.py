"""Single-worker-per-GPU durable queue; preserves failures and all run outputs."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--queue',default='revision/queue.json');a=p.parse_args()
q=Path(a.queue).resolve();lock=q.with_suffix('.lock');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu;env['OMP_NUM_THREADS']='4';env['HF_HOME']=str(q.parents[2]/'hf_cache');env['TOKENIZERS_PARALLELISM']='false'
while True:
    with lock.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());job=next((j for j in jobs if j['status']=='pending'),None)
        if job is None:break
        job.update(status='running',gpu=a.gpu,worker_pid=os.getpid(),started=time.time());q.write_text(json.dumps(jobs,indent=2));chosen=dict(job)
    out=Path(chosen['out']);out.mkdir(parents=True,exist_ok=True)
    with (out/'console.log').open('a') as log:
        result=subprocess.run([sys.executable]+chosen['args'],env=env,stdout=log,stderr=subprocess.STDOUT)
    with lock.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());job=next(j for j in jobs if j['id']==chosen['id']);job.update(status='complete' if result.returncode==0 and (out/'summary.json').exists() else 'failed',returncode=result.returncode,ended=time.time());q.write_text(json.dumps(jobs,indent=2))
