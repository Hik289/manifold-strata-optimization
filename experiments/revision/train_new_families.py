"""Prospective real-data tasks, paired seed42, complete 1000-epoch budgets.
CPU/GPU backends are explicit. No candidate is chosen using test measurements.
"""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stream import optimizer,flag_snapshot,tensor_state_bytes

class ForecastMLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(96*7,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,24*7))
 def forward(self,x):return self.net(x.flatten(1))
class RatingMF(nn.Module):
 def __init__(self,users,items,mean):
  super().__init__();self.users=nn.Embedding(users,128);self.items=nn.Embedding(items,128);self.ub=nn.Embedding(users,1);self.ib=nn.Embedding(items,1);self.register_buffer('mean',torch.tensor(mean));nn.init.normal_(self.users.weight,std=.05);nn.init.normal_(self.items.weight,std=.05);nn.init.zeros_(self.ub.weight);nn.init.zeros_(self.ib.weight)
 def forward(self,x):return self.mean+self.ub(x[:,0]).squeeze(-1)+self.ib(x[:,1]).squeeze(-1)+(self.users(x[:,0])*self.items(x[:,1])).sum(-1)/math.sqrt(128)
class GasMLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6))
 def forward(self,x):return self.net(x)

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['ETTh1','ETTh2','ml100k','ml1m','gas_drift']);p.add_argument('--base',default='static',choices=['static','finetune','replay']);p.add_argument('--method',required=True,choices=['strata','single','adamw','galore','independent']);p.add_argument('--lr',type=float,required=True);p.add_argument('--flag-xi',type=float);p.add_argument('--rank',type=int,default=32);p.add_argument('--epochs',type=int,default=1000);p.add_argument('--batch',type=int,default=512);p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert a.smoke or a.epochs==1000;assert (a.task=='gas_drift')==(a.base in ['finetune','replay'])
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42);dev=torch.device(a.device);root=Path(a.out);root.mkdir(parents=True,exist_ok=True);(root/'flags').mkdir(exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True)
 if (root/'summary.json').exists():return
 source=Path(a.data)/'new_families';meta=json.loads((source/(a.task+'.json')).read_text());raw=np.load(source/(a.task+'.npz'));assert hashlib.sha256((source/(a.task+'.npz')).read_bytes()).hexdigest()==meta['array_sha256'];data={};bounds=None
 if a.task.startswith('ETT'):
  seq=raw['series'];lookback=96;horizon=24
  for split,(lo,hi) in meta['bounds'].items():
   targets=np.arange(max(lo,lookback),hi-horizon+1);xx=np.stack([seq[t-lookback:t] for t in targets]);yy=np.stack([seq[t:t+horizon].reshape(-1) for t in targets]);assert targets.min()>=lo and targets.max()+horizon<=hi
   if a.smoke:xx=xx[:128];yy=yy[:128]
   data[split]=(torch.tensor(xx,device=dev),torch.tensor(yy,device=dev))
  model=ForecastMLP();metric='MSE';direction='min';modelname='MLP256-L96-H24';dataset=a.task+'_forecast24';family='forecasting'
 elif a.task.startswith('ml'):
  for split in ['train','validation','test']:
   xx=raw[split+'_ui'];yy=raw[split+'_y']
   if a.smoke:xx=xx[:256];yy=yy[:256]
   data[split]=(torch.tensor(xx,device=dev),torch.tensor(yy,device=dev))
  model=RatingMF(meta['n_users'],meta['n_items'],meta['rating_train_mean']);metric='RMSE';direction='min';modelname='BiasedMF128';dataset=a.task+'_ratings';family='recommendation'
 else:
  gx=torch.tensor(raw['x'],device=dev);gy=torch.tensor(raw['y'],device=dev);gb=torch.tensor(raw['batch'],device=dev);gr=torch.tensor(raw['role'],device=dev);train_by=[torch.where((gr==0)&(gb==k))[0] for k in range(10)]
  if a.smoke:train_by=[idx[:64] for idx in train_by]
  model=GasMLP();metric='acc';direction='max';modelname='MLP256-domains10';dataset='gas_drift';family='sensor_drift'
 model=model.to(dev);opt=optimizer(model,a);prev={};step=0;start=0;domain_length=1 if a.smoke else 100
 info=vars(a)|dict(dataset=dataset,model=modelname,family=family,seed=42,ablation='prospective_grid',primary_metric=metric,primary_direction=direction,split=meta,precision='float32',backend=str(dev),selection='mean primary validation metric over terminal100 epochs; all five methods receive six predeclared candidates',epoch_definition='one complete pass over the current domain training set' if a.task=='gas_drift' else 'one complete pass over the full training set',schedule='cosine with 5-epoch warmup; restart per domain only for drift',replay_capacity=512 if a.base=='replay' else 0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),optimizer_source_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'strata_pkg/strata_adamw.py').read_bytes()).hexdigest())
 (root/'config.json').write_text(json.dumps(info,indent=2));latest=root/'checkpoints/latest.pt'
 if latest.exists():
  ck=torch.load(latest,map_location=dev,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch'];step=ck['step'];log=root/'log.jsonl'
  if log.exists():
   rows=log.read_text().splitlines();tail=[s for s in rows if json.loads(s)['epoch']>start]
   if tail:(root/f'resume_tail_after_{start}_{int(time.time())}.jsonl').write_text('\n'.join(tail)+'\n');log.write_text('\n'.join(s for s in rows if json.loads(s)['epoch']<=start)+'\n')
 else:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=0,step=0),root/'checkpoints/epoch_0000.pt')
 def lossfun(z,y):return F.cross_entropy(z,y) if a.task=='gas_drift' else F.mse_loss(z,y)
 @torch.no_grad()
 def evaluate(split,task):
  model.eval()
  if a.task=='gas_drift':
   mask=(gr==({'validation':1,'test':2}[split]))&(gb<=task);idx=torch.where(mask)[0]
   if a.smoke:idx=torch.cat([torch.where(mask&(gb==k))[0][:64] for k in range(task+1)])
   losses=torch.zeros(task+1,device=dev);correct=torch.zeros_like(losses);counts=torch.zeros_like(losses)
   for ix in idx.split(2048):
    z=model(gx[ix]);ll=F.cross_entropy(z,gy[ix],reduction='none');dd=gb[ix];losses.scatter_add_(0,dd,ll);correct.scatter_add_(0,dd,(z.argmax(-1)==gy[ix]).float());counts.scatter_add_(0,dd,torch.ones_like(ll))
   assert bool((counts>0).all());acc=correct/counts;ls=losses/counts;return dict(loss=float(ls.mean()),metric=float(acc.mean()),per_domain_acc=acc.cpu().tolist())
  x,y=data[split];sse=sae=total=0
  for idx in torch.arange(len(x),device=dev).split(2048):
   z=model(x[idx]);err=z-y[idx];sse+=float((err*err).sum());sae+=float(err.abs().sum());total+=err.numel()
  mse=sse/total;return dict(loss=mse,metric=mse if metric=='MSE' else math.sqrt(mse),mae=sae/total)
 for ep in range(start+1,a.epochs+1):
  torch.manual_seed(42+ep);rng=torch.Generator(device=dev).manual_seed(42+ep);t0=time.monotonic();model.train();task=min(9,(ep-1)//domain_length) if a.task=='gas_drift' else 0;span=domain_length if a.task=='gas_drift' else a.epochs;within=(ep-1)%span+1;lr=a.lr*min(1,within/5)*.5*(1+math.cos(math.pi*(within-1)/span));logs=[];ls=total=gn=nb=0
  for group in opt.param_groups:group['lr']=lr
  if dev.type=='cuda':torch.cuda.reset_peak_memory_stats()
  if a.task=='gas_drift':
   train=train_by[task];order=train[torch.randperm(len(train),generator=rng,device=dev)];past=torch.cat(train_by[:task]) if task else train[:0];mrng=torch.Generator(device=dev).manual_seed(4242+task);memory=past[torch.randperm(len(past),generator=mrng,device=dev)[:512]] if len(past) else past
  else:
   x,y=data['train'];order=torch.randperm(len(x),generator=rng,device=dev)
  for idx in order.split(a.batch):
   if a.task=='gas_drift':
    if a.base=='replay' and len(memory):idx=torch.cat([idx,memory[torch.randperm(len(memory),generator=rng,device=dev)[:128]]])
    xx,yy=gx[idx],gy[idx]
   else:xx,yy=x[idx],y[idx]
   opt.zero_grad(set_to_none=True);z=model(xx);loss=lossfun(z,yy)
   if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite train loss at epoch{ep} step{step}')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step();step+=1;gn+=float(norm);nb+=1;ls+=float(loss.detach())*len(idx);total+=len(idx)
   if any(s.get('t',-1)%50==1 for s in opt.state.values() if 'U' in s):logs.append(flag_snapshot(opt,root/'flags',step,prev))
  val=evaluate('validation',task);test=evaluate('test',task);row=dict(epoch=ep,step=step,domain=task+1 if a.task=='gas_drift' else None,train_loss=ls/total,val_loss=val['loss'],test_loss=test['loss'],val_metric=val['metric'],test_metric=test['metric'],metric=metric,gradient_norm=gn/max(1,nb),lr=lr,wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated() if dev.type=='cuda' else 0,optimizer_state_bytes=tensor_state_bytes(opt),train_examples=total,flag_diagnostics=logs,validation=val,test=test)
  if metric=='acc':row.update(val_acc=val['metric'],test_acc=test['metric'])
  assert all(math.isfinite(row[k]) for k in ['train_loss','val_loss','test_loss','val_metric','test_metric']),row
  with (root/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if ep%10==0 or ep in [1,2,5,a.epochs]:
   ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep,step=step);torch.save(ck,root/'checkpoints'/f'epoch_{ep:04d}.pt');tmp=latest.with_suffix('.partial');torch.save(ck,tmp);tmp.replace(latest)
  if ep%25==0 or a.smoke:print(json.dumps({k:row[k] for k in ['epoch','domain','train_loss','val_metric','test_metric','wall_seconds']}),flush=True)
 (root/'summary.json').write_text(json.dumps(dict(complete=True,smoke=a.smoke,final=row),indent=2))
if __name__=='__main__':main()
