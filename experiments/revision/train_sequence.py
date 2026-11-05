"""Full-corpus recurrent/Transformer comparisons; one epoch visits all train tokens."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent));from train_stream import optimizer,flag_snapshot
class LM(nn.Module):
 def __init__(self,kind,n):
  super().__init__();self.kind=kind;self.emb=nn.Embedding(n,256);self.drop=nn.Dropout(.2);self.head=nn.Linear(256,n)
  if kind=='transformer':self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,4,1024,.2,batch_first=True),2,enable_nested_tensor=False);self.pos=nn.Parameter(torch.randn(1,128,256)*.02)
  else:self.body=(nn.LSTM if kind=='lstm' else nn.GRU)(256,256,2,dropout=.2,batch_first=True)
 def forward(self,x):
  x=self.drop(self.emb(x))
  if self.kind=='transformer':x=self.body(x+self.pos[:,:x.shape[1]],mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1],device=x.device),is_causal=True)
  else:x=self.body(x)[0]
  return self.head(self.drop(x))
p=argparse.ArgumentParser();p.add_argument('--model',choices=['lstm','gru','transformer'],required=True);p.add_argument('--method',required=True);p.add_argument('--rank',type=int,default=64);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--epochs',type=int,default=1000);p.add_argument('--batch',type=int,default=32);p.add_argument('--length',type=int,default=128);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert a.smoke or a.epochs>=1000
torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);root=Path(a.out);root.mkdir(parents=True,exist_ok=True);(root/'flags').mkdir(exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True)
if (root/'summary.json').exists():sys.exit(0)
meta=json.loads((Path(a.data)/'sequence/wikitext2_words.json').read_text());raw=np.load(Path(a.data)/'sequence/wikitext2_words.npz');data={s:torch.tensor(v.copy(),dtype=torch.long) for s,v in raw.items()}
if a.smoke:data={s:v[:a.length*4+1] for s,v in data.items()}
model=LM(a.model,meta['vocabulary']).cuda();opt=optimizer(model,a);info=vars(a)|dict(seed=42,dataset='wikitext2_words',split=meta,precision='BF16 autocast, FP32 model/state',epoch_definition='all official train tokens in contiguous nonoverlapping next-token blocks; recurrent hidden state reset per block',device=torch.cuda.get_device_name(0),parameters=sum(v.numel() for v in model.parameters()),selection='terminal-100 validation cross-entropy',token_budget_per_epoch=len(data['train'])-1);(root/'config.json').write_text(json.dumps(info,indent=2));step=0;best=float('inf');prev={};start=0
latest=root/'checkpoints/latest.pt'
if latest.exists():
 c=torch.load(latest,map_location='cuda',weights_only=False);model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);start=c['epoch'];step=c['step'];best=c['best']
else:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=0),root/'checkpoints/epoch_0000.pt')
def batches(s):
 v=data[s];nfull=(len(v)-1)//a.length;starts=list(range(0,nfull*a.length,a.length))
 for j in range(0,len(starts),a.batch):
  ss=starts[j:j+a.batch];xx=torch.stack([v[i:i+a.length] for i in ss]).cuda();yy=torch.stack([v[i+1:i+a.length+1] for i in ss]).cuda();yield xx,yy
 offset=nfull*a.length
 if offset<len(v)-1:yield v[offset:-1][None].cuda(),v[offset+1:][None].cuda()
@torch.no_grad()
def evaluate(s):
 model.eval();loss=total=0
 for x,y in batches(s):
  with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x);ls=F.cross_entropy(z.reshape(-1,meta['vocabulary']),y.reshape(-1),reduction='sum')
  loss+=float(ls);total+=y.numel()
 return loss/total
for ep in range(start+1,a.epochs+1):
 torch.manual_seed(42+ep);model.train();t0=time.monotonic();ls=nt=gn=steps=0;logs=[];lr=a.lr*min(1,ep/10)*.5*(1+math.cos(math.pi*(ep-1)/a.epochs));torch.cuda.reset_peak_memory_stats()
 for g in opt.param_groups:g['lr']=lr
 for x,y in batches('train'):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x);loss=F.cross_entropy(z.reshape(-1,meta['vocabulary']),y.reshape(-1))
  loss.backward();gn+=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step();step+=1;steps+=1;ls+=float(loss)*y.numel();nt+=y.numel()
  if any(s.get('t',-1)%50==1 for s in opt.state.values() if 'U' in s):logs.append(flag_snapshot(opt,root/'flags',step,prev))
 vl=evaluate('validation');tl=evaluate('test');row=dict(epoch=ep,tokens=nt*ep,train_loss=ls/nt,val_loss=vl,test_loss=tl,val_ppl=math.exp(min(vl,50)),test_ppl=math.exp(min(tl,50)),gradient_norm=gn/max(1,steps),lr=lr,wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated(),flag_diagnostics=logs)
 assert all(math.isfinite(row[k]) for k in ['train_loss','val_loss','test_loss']),row
 with (root/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if vl<best:best=vl;torch.save(dict(model=model.state_dict(),epoch=ep,val_loss=vl),root/'checkpoints/best_val.pt')
 if ep%10==0 or ep in [1,2,5,20,50,100,200,500,1000]:
  c=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep,step=step,best=best);torch.save(c,root/'checkpoints'/f'epoch_{ep:04d}.pt');torch.save(c,latest)
 print(json.dumps({k:row[k] for k in ['epoch','train_loss','val_ppl','wall_seconds']}),flush=True)
(root/'summary.json').write_text(json.dumps(dict(complete=True,best_val=best,final=row),indent=2))
