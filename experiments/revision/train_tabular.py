"""Full-data, paired seed-42 UCI MLPs; 1000 epochs for each formal trial."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stream import optimizer,flag_snapshot
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['adult','wine_red','wine_white','covertype']);p.add_argument('--width',type=int,default=256);p.add_argument('--method',required=True);p.add_argument('--rank',type=int,default=64);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--epochs',type=int,default=1000);p.add_argument('--batch',type=int,default=512);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--ablation',default='full',choices=['full','same_rho','reverse_rho','equal_a','no_residual','random_flag','no_transport','frozen_after_warmup','heterogeneous_memory','periodic_svd','least_squares_transport']);p.add_argument('--flag-warmup-steps',type=int,default=200);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert a.smoke or a.epochs>=1000
torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);root=Path(a.out);root.mkdir(parents=True,exist_ok=True);(root/'flags').mkdir(exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True)
if (root/'summary.json').exists():sys.exit(0)
meta=json.loads((Path(a.data)/'tabular'/f'{a.dataset}.json').read_text());d=np.load(Path(a.data)/'tabular'/f'{a.dataset}.npz');data={s:(torch.tensor(d[s+'_x'],device='cuda'),torch.tensor(d[s+'_y'],device='cuda')) for s in ['train','validation','test']}
if a.smoke:data={s:(x[:512],y[:512]) for s,(x,y) in data.items()}
model=nn.Sequential(nn.Linear(meta['n_features'],a.width),nn.ReLU(),nn.Linear(a.width,a.width),nn.ReLU(),nn.Linear(a.width,meta['n_classes'])).cuda();opt=optimizer(model,a);info=vars(a)|dict(seed=42,model=f'MLP-{a.width}-{a.width}',split=meta,precision='float32',schedule='1000 epoch cosine; full datasets',device=torch.cuda.get_device_name(0),selection='terminal-100 validation accuracy across predeclared learning rates')
(root/'config.json').write_text(json.dumps(info,indent=2));best=-1.;step=0;prev={};start=0
ck=root/'checkpoints/latest.pt'
if ck.exists():
 c=torch.load(ck,map_location='cuda',weights_only=False);model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);start=c['epoch'];step=c['step'];best=c['best']
else:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=0),root/'checkpoints/epoch_0000.pt')
@torch.no_grad()
def evaluate(s):
 model.eval();x,y=data[s];ls=correct=0
 for ix in torch.arange(len(x),device='cuda').split(8192):
  z=model(x[ix]);ls+=float(F.cross_entropy(z,y[ix],reduction='sum'));correct+=int((z.argmax(1)==y[ix]).sum())
 return ls/len(x),correct/len(x)
for ep in range(start+1,a.epochs+1):
 torch.manual_seed(42+ep);t0=time.monotonic();model.train();x,y=data['train'];order=torch.randperm(len(x),device='cuda');ls=gn=total=0.;lr=a.lr*.5*(1+math.cos(math.pi*(ep-1)/a.epochs));logs=[];torch.cuda.reset_peak_memory_stats()
 for g in opt.param_groups:g['lr']=lr
 for ix in order.split(a.batch):
  opt.zero_grad(set_to_none=True);z=model(x[ix]);loss=F.cross_entropy(z,y[ix]);loss.backward();gn+=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.));opt.step();step+=1;ls+=float(loss)*len(ix);total+=len(ix)
  if any(s.get('t',-1)%50==1 for s in opt.state.values() if 'U' in s):logs.append(flag_snapshot(opt,root/'flags',step,prev))
 vl,va=evaluate('validation');tl,ta=evaluate('test');row=dict(epoch=ep,train_loss=ls/total,val_loss=vl,val_acc=va,test_loss=tl,test_acc=ta,gradient_norm=gn/math.ceil(len(x)/a.batch),lr=lr,wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated(),flag_diagnostics=logs)
 with (root/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if va>best:best=va;torch.save(dict(model=model.state_dict(),epoch=ep,val_acc=va),root/'checkpoints/best_val.pt')
 if ep%10==0 or ep in [1,2,5,20,50,100,200,500,1000]:
  c=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep,step=step,best=best);torch.save(c,root/'checkpoints'/f'epoch_{ep:04d}.pt');torch.save(c,ck)
 if ep%25==0 or a.smoke:print(json.dumps({k:row[k] for k in ['epoch','train_loss','val_acc','test_acc','wall_seconds']}),flush=True)
(root/'summary.json').write_text(json.dumps(dict(complete=True,best_val=best,final=row),indent=2))
