"""Validate all optimizer/model paths before releasing the new full-budget matrix."""
import json,os,subprocess,sys
from pathlib import Path
root=Path('..').resolve();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4')
smokes=[]
for method in ['strata','single','adamw','galore','independent']:
 smokes.append(['revision/train_tabular.py','--dataset','wine_red','--method',method,'--epochs','1','--smoke','--data','../data','--out',f'smoke_breadth/wine_red_{method}'])
for ds in ['adult','wine_white','covertype']:
 smokes.append(['revision/train_tabular.py','--dataset',ds,'--method','strata','--width','512','--epochs','1','--smoke','--data','../data','--out',f'smoke_breadth/{ds}_strata512'])
for model in ['lstm','gru','transformer']:
 for method in ['strata','single','adamw','galore','independent']:
  smokes.append(['revision/train_sequence.py','--model',model,'--method',method,'--epochs','1','--smoke','--data','../data','--out',f'smoke_breadth/{model}_{method}'])
for args in smokes:
 print(json.dumps(dict(smoke=args)),flush=True);subprocess.run([sys.executable]+args,env=env,check=True)
subprocess.run([sys.executable,'revision/enqueue_breadth.py'],check=True)
settings=[('2','ablation'),('3','tabular'),('3','sequence')];pids={}
for gpu,families in settings:
 key=f'{gpu}_{families}';log=(root/f'worker_breadth_{key}.log').open('a')
 proc=subprocess.Popen([sys.executable,'-u','revision/run_breadth_queue.py','--gpu',gpu,'--families',families,'--watch'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);pids[key]=proc.pid
(root/'worker_breadth_pids.json').write_text(json.dumps(pids,indent=2));print(json.dumps(dict(all_smokes_passed=len(smokes),workers=pids)),flush=True)
