"""Generated Darcy operator benchmark, not a claim to an external FNO dataset.
Variable-coefficient elliptic PDE, zero Dirichlet boundary, unit forcing.
Finite-volume fluxes use harmonic face coefficients. Every split is generated
before model fitting with a fixed RNG, and the exact arrays are retained.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stream import optimizer,flag_snapshot
class Spectral(nn.Module):
 def __init__(self,width=96,modes=4):
  super().__init__();self.m=modes;self.w=nn.Parameter(torch.randn(2,width,width,modes,modes,2)*.01)
 def forward(self,x):
  z=torch.fft.rfft2(x);out=z.new_zeros(z.shape);m=self.m
  for k,s in enumerate([slice(0,m),slice(-m,None)]):out[:,:,s,:m]=torch.einsum('bixy,ioxy->boxy',z[:,:,s,:m],torch.view_as_complex(self.w[k].contiguous()))
  return torch.fft.irfft2(out,s=x.shape[-2:])
class FNO(nn.Module):
 def __init__(self):
  super().__init__();self.lift=nn.Conv2d(1,96,1);self.spectral=nn.ModuleList([Spectral() for _ in range(4)]);self.local=nn.ModuleList([nn.Conv2d(96,96,1) for _ in range(4)]);self.head=nn.Sequential(nn.Conv2d(96,128,1),nn.GELU(),nn.Conv2d(128,1,1))
 def forward(self,x):
  x=self.lift(x)
  for spec,local in zip(self.spectral,self.local):x=F.gelu(spec(x)+local(x))
  return self.head(x)
def generate(n=1200,side=16):
 rng=np.random.default_rng(42);xs=[];ys=[];h=1/(side+1)
 for _ in range(n):
  z=gaussian_filter(rng.normal(size=(side,side)),1.5);z=z/max(z.std(),1e-8);a=np.exp(.5*z);r=[];c=[];v=[]
  for i in range(side):
   for j in range(side):
    idx=i*side+j;diag=0.
    for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
     ni,nj=i+di,j+dj
     if 0<=ni<side and 0<=nj<side:
      face=2*a[i,j]*a[ni,nj]/(a[i,j]+a[ni,nj]);r.append(idx);c.append(ni*side+nj);v.append(-face/h**2);diag+=face/h**2
     else:diag+=a[i,j]/h**2
    r.append(idx);c.append(idx);v.append(diag)
  mat=coo_matrix((v,(r,c)),shape=(side*side,side*side)).tocsr();u=spsolve(mat,np.ones(side*side)).reshape(side,side);xs.append(a);ys.append(u)
 return np.array(xs,dtype=np.float32)[:,None],np.array(ys,dtype=np.float32)[:,None]
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--lr',type=float,default=.001);p.add_argument('--rank',type=int,default=64);p.add_argument('--epochs',type=int,default=2000);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert a.epochs>=2000 or a.smoke
 torch.set_num_threads(4);torch.manual_seed(42);cache=Path(a.data)/'darcy_generated_seed42.npz'
 if not cache.exists():
  x,y=generate();np.savez_compressed(cache,x=x,y=y)
 ds=np.load(cache);x=torch.tensor(ds['x'],device='cuda');y=torch.tensor(ds['y'],device='cuda');scale=y[:1000].std();y=y/scale
 model=FNO().cuda();opt=optimizer(model,a);root=Path(a.out);root.mkdir(parents=True,exist_ok=True);(root/'flags').mkdir(exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True);(root/'config.json').write_text(json.dumps(vars(a)|dict(seed=42,split=[1000,100,100],grid=16,operator='-div(a grad u)=1; Dirichlet0; harmonic-face finite volume',normalization='training target standard deviation',spectral_optimizer='real/imag 6D tensor uses full Adam branch; pointwise matrices eligible for flags'),indent=2));best=float('inf');prev={};step=0
 @torch.no_grad()
 def ev(s):
  model.eval();z=model(x[s]);target=y[s];err=z-target;spec=torch.fft.rfft2(err);freq=spec.abs().square()
  u=z*scale;coef=x[s];up=F.pad(u,(1,1,1,1));ap=F.pad(coef,(1,1,1,1),mode='replicate');n=u.shape[-1];res=torch.zeros_like(u)
  for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
   un=up[:,:,1+di:1+di+n,1+dj:1+dj+n];an=ap[:,:,1+di:1+di+n,1+dj:1+dj+n];face=2*coef*an/(coef+an);res+=face*(u-un)*(n+1)**2
  residual=float((res-1).square().mean())
  return dict(pde_residual=residual,rel_l2=float((err.flatten(1).norm(dim=1)/target.flatten(1).norm(dim=1)).mean()),loss=float(err.square().mean()),low_freq_error=float(freq[:,:,:4,:4].mean()),high_freq_error=float(freq[:,:,4:,4:].mean()))
 for ep in range(1,a.epochs+1):
  t0=time.monotonic();model.train();total=gn=0.;logs=[];ids=torch.randperm(1000,device='cuda')
  for idx in ids.split(100):
   opt.zero_grad(set_to_none=True);z=model(x[idx]);loss=(z-y[idx]).square().mean();loss.backward();gn+=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.));opt.step();step+=1;total+=float(loss)
   if step%50==1:logs.append(flag_snapshot(opt,root/'flags',step,prev))
  val=ev(slice(1000,1100));test=ev(slice(1100,1200));row=dict(epoch=ep,train_loss=total/10,val=val,test=test,gradient_norm=gn/10,wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated(),lr=a.lr,flag_diagnostics=logs)
  with (root/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if val['rel_l2']<best:best=val['rel_l2'];torch.save(dict(model=model.state_dict(),epoch=ep),root/'checkpoints'/'best_val.pt')
  if ep%10==0 or ep in [1,2,5,20,50,100,200,500,1000]:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep),root/'checkpoints'/f'epoch_{ep:04d}.pt')
  if ep%10==0:print(json.dumps({k:row[k] for k in ['epoch','val','test','wall_seconds']}),flush=True)
 (root/'summary.json').write_text(json.dumps(dict(complete=True,best_val=best,final=row),indent=2))
if __name__=='__main__':main()
