"""Full-data, seed-42 continual/nonstationary benchmarks; no test-based selection.
Run from experiments/. Formal runs require 1000 epochs. Evaluation covers all
previous tasks for CL and the current distribution for shifts.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from strata_pkg.strata_adamw import STRATAAdamW
from strata_pkg.baselines import SingleGrassmannAdamW, GaLoreAdamW, IndependentGrassmannAdamW

def optimizer(model,a):
    kw=dict(lr=a.lr,weight_decay=0.,seed=42)
    ab=getattr(a,'ablation','full')
    if a.method=='adamw': return torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.,eps=1e-6)
    if a.method=='galore': return GaLoreAdamW(model.parameters(),lr=a.lr,R=a.rank,tau=50,scale=1.,weight_decay=0.,eps=1e-6)
    if a.method=='single': return SingleGrassmannAdamW(model.parameters(),R=a.rank,rho=.999,a=1.,a_perp=.2,tau=50,eps=1e-6,**kw)
    if a.method=='independent': return IndependentGrassmannAdamW(model.parameters(),ranks=[a.rank//4,a.rank//2,a.rank],rhos=[.999,.98,.9],a_perp=.2,tau=50,eps=1e-6,**kw)
    opt=STRATAAdamW(model.parameters(),ranks=[a.rank//4,a.rank//2,a.rank],rhos=[.98]*3 if ab=='same_rho' else [.9,.98,.999] if ab=='reverse_rho' else [.999,.98,.9],a_ks=[1.]*3 if ab=='equal_a' else [1.,.7,.4],beta1s=[.99,.9,.5] if ab=='heterogeneous_memory' else [.9]*3,beta2s=[.999,.99,.9] if ab=='heterogeneous_memory' else [.999]*3,a_perp=0. if ab=='no_residual' else 1. if ab=='equal_a' else .2,tau=50,xi=0. if ab=='random_flag' else 1e-3,eps=1e-6,**kw)
    if ab in ['no_transport','frozen_after_warmup','least_squares_transport','periodic_svd']:
        original=opt._flag_update
        def flag_update(state,work,group):
            if ab=='frozen_after_warmup' and state['t']>=getattr(a,'flag_warmup_steps',10000): return
            old=[v.clone() for v in state['M']];oldU=state['U'].clone()
            if ab=='periodic_svd':
                from strata_pkg import geometry as geo
                eye=torch.eye(state['d'],device=work.device,dtype=work.dtype)
                cov=sum(w*geo.nystrom_apply(y,o,eye,group['gamma']) for w,y,o in zip(group['omegas'],state['Y'],state['Omega']))
                _,ev=torch.linalg.eigh((cov+cov.T)/2);bar=ev[:,-group['ranks'][-1]:].flip(1);new=torch.empty_like(oldU)
                for k,sl in enumerate(state['slices']):
                    O=geo.procrustes(bar[:,sl],oldU[:,sl]);new[:,sl]=bar[:,sl]@O;state['M'][k]=old[k]@O.T;state['V'][k].zero_();state['c_V'][k]=0
                state['U']=new
                return
            original(state,work,group)
            if ab=='no_transport':state['M']=old
            if ab=='least_squares_transport':state['M']=[m@oldU[:,sl].T@state['U'][:,sl] for m,sl in zip(old,state['slices'])]
        opt._flag_update=flag_update
    return opt

class MLP(nn.Module):
    def __init__(self,d,n):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(d,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,n))
    def forward(self,x): return self.net(x)

@torch.no_grad()
def rotate(x,angle):
    if not angle: return x
    r=math.radians(angle); theta=x.new_tensor([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0]])
    grid=F.affine_grid(theta[None].expand(len(x),-1,-1),x.shape,align_corners=False)
    return F.grid_sample(x,grid,align_corners=False)

@torch.no_grad()
def evaluate(model,x,y,angle=0,weights=None):
    model.eval(); loss=correct=denom=0
    for j in range(0,len(x),1024):
        z=model(rotate(x[j:j+1024],angle)); ww=torch.ones(len(z),device=z.device) if weights is None else weights[j:j+1024]
        loss+=(F.cross_entropy(z,y[j:j+1024],reduction='none')*ww).sum().item(); correct+=((z.argmax(1)==y[j:j+1024])*ww).sum().item(); denom+=ww.sum().item()
    return dict(loss=loss/denom,acc=correct/denom)

def tensor_state_bytes(opt):
    seen=set()
    def count(v):
        if torch.is_tensor(v):
            key=(str(v.device),v.untyped_storage().data_ptr())
            if key in seen: return 0
            seen.add(key); return v.untyped_storage().nbytes()
        if isinstance(v,dict): return sum(count(x) for x in v.values())
        if isinstance(v,(list,tuple)): return sum(count(x) for x in v)
        return 0
    return count(opt.state)

def flag_snapshot(opt,folder,step,prev):
    data={}; energies=[]; drifts=[]; orth=[]; angles=[]; recon=[]
    for i,(p,s) in enumerate(opt.state.items()):
        if 'U' not in s: continue
        if isinstance(s['U'],list):
            for k,basis in enumerate(s['U']):
                u=basis.detach();key=f'{i}/{k}';data[key]={'U':u.cpu().half(),'t':s['t']};orth.append(float((u.T@u-torch.eye(u.shape[1],device=u.device)).norm()))
                if key in prev:
                    old=prev[key].to(u.device);cross=old.T@u;drifts.append(float((2*u.shape[1]-2*cross.square().sum()).clamp_min(0).sqrt()));angles.append([float(torch.acos(torch.linalg.svdvals(cross).min().clamp(-1,1))*180/math.pi)])
                prev[key]=u.cpu().clone()
            continue
        u=s['U'].detach(); data[str(i)]={'U':u.cpu().half(),'t':s['t']}
        orth.append(float((u.T@u-torch.eye(u.shape[1],device=u.device)).norm()))
        stats=s.get('last_stats',{}); energies.append(stats)
        if i in prev:
            old=prev[i].to(u.device); cross=old.T@u
            drifts.append(float((2*u.shape[1]-2*cross.square().sum()).clamp_min(0).sqrt()))
            cuts=[z.stop for z in s.get('slices',[slice(0,u.shape[1])])]
            angles.append([float(torch.acos(torch.linalg.svdvals(old[:,:r].T@u[:,:r]).min().clamp(-1,1))*180/math.pi) for r in cuts])
        if 'level_energy' in stats:
            en=stats['level_energy']; resid=stats['residual_energy'];recon.append([resid+sum(en[k+1:]) for k in range(len(en))])
        prev[i]=u.cpu().clone()
    if data: torch.save(data,folder/f'flag_{step:09d}.pt')
    return dict(energy=energies,projector_drift=drifts,orthogonality=orth,principal_angle_deg=angles,reconstruction_energy=recon,optimizer_state_bytes=tensor_state_bytes(opt))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['rotated_mnist_cl','rotated_mnist_shift','split_cifar100','cifar10_label_shift','cifar100_class_shift'],required=True)
    p.add_argument('--method',choices=['adamw','galore','single','independent','strata'],required=True)
    p.add_argument('--base',choices=['finetune','replay','ewc'],default='finetune'); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--rank',type=int,default=64)
    p.add_argument('--epochs',type=int,default=1000); p.add_argument('--batch',type=int,default=512); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--smoke',action='store_true')
    p.add_argument('--ablation',choices=['full','same_rho','reverse_rho','equal_a','no_residual','random_flag','no_transport','frozen_after_warmup','heterogeneous_memory','periodic_svd','least_squares_transport'],default='full')
    a=p.parse_args(); assert a.epochs>=1000 or a.smoke
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    root=Path(a.out); root.mkdir(parents=True,exist_ok=True); (root/'flags').mkdir(exist_ok=True); (root/'checkpoints').mkdir(exist_ok=True)
    if (root/'summary.json').exists(): return
    device=torch.device('cuda'); cls=MNIST if 'mnist' in a.task else CIFAR100 if '100' in a.task else CIFAR10
    ds=cls(a.data,train=True,download=True); ts=cls(a.data,train=False,download=True)
    def tensors(d):
        x=torch.as_tensor(np.array(d.data),device=device).float()/255
        x=x[:,None] if x.ndim==3 else x.permute(0,3,1,2)
        return x,torch.as_tensor(d.targets,device=device).long()
    x,y=tensors(ds); xt,yt=tensors(ts); perm=torch.randperm(len(x),generator=torch.Generator().manual_seed(42)).to(device)
    vi=perm[:5000]; ti=perm[5000:]; xv,yv=x[vi],y[vi]; x,y=x[ti],y[ti]
    if a.smoke: x,y=x[:1024],y[:1024]; xv,yv=xv[:256],yv[:256]; xt,yt=xt[:256],yt[:256]
    if 'cifar' in a.task:
        from torchvision.models import resnet18
        model=resnet18(num_classes=100 if cls==CIFAR100 else 10); model.conv1=nn.Conv2d(3,64,3,1,1,bias=False); model.maxpool=nn.Identity(); model=model.to(device)
    else: model=MLP(784,10).to(device)
    opt=optimizer(model,a); cfg=vars(a)|dict(seed=42,selection='highest mean seen-task validation accuracy; test never used',full_train_examples=len(x),validation_examples=len(xv),test_examples=len(xt),architecture='MLP-256-256' if cls==MNIST else 'ResNet18-CIFAR',eps=1e-6)
    (root/'config.json').write_text(json.dumps(cfg,indent=2)); n_tasks=10 if 'mnist' in a.task or a.task=='split_cifar100' else 4
    best=-1; best_epoch=0; seen_best=np.zeros(n_tasks); prev={}; step=0; last=[]; ewc=[]; replay_ids=torch.empty(0,dtype=torch.long,device=device); replay_angles=torch.empty(0,device=device)
    start_epoch=0; resume=root/'checkpoints'/'latest.pt'
    if resume.exists():
        ck=torch.load(resume,map_location=device,weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start_epoch=ck['epoch']; step=ck['step']; best=ck['best']; seen_best=ck['seen_best']; replay_ids=ck['replay_ids']; ewc=ck['ewc']; replay_angles=ck['replay_angles']
    if start_epoch==0: torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=0,step=0),root/'checkpoints'/'epoch_0000.pt')
    for ep in range(start_epoch+1,a.epochs+1):
        torch.manual_seed(42+ep); t0=time.monotonic(); task=min(n_tasks-1,(ep-1)*n_tasks//a.epochs); angle=task*10 if cls==MNIST else 0
        ids=torch.arange(len(x),device=device)
        if a.task=='split_cifar100': ids=ids[(y>=task*10)&(y<(task+1)*10)]
        if a.task=='cifar100_class_shift': ids=ids[(y>=task*25)&(y<(task+1)*25)]
        if a.task=='cifar10_label_shift':
            weight=torch.where(y%4==task,torch.ones_like(y,dtype=torch.float),torch.full_like(y,.1,dtype=torch.float)); ids=torch.multinomial(weight,len(x),replacement=True)
        ids=ids[torch.randperm(len(ids),device=device)]; model.train(); loss_sum=correct=total=gn=0.; diagnostics=[]
        torch.cuda.reset_peak_memory_stats()
        for idx in ids.split(a.batch):
            xb,yb=rotate(x[idx],angle),y[idx]
            if a.base=='replay' and len(replay_ids):
                pos=torch.randint(len(replay_ids),(max(1,len(idx)//4),),device=device); ri=replay_ids[pos]; rx=x[ri].clone()
                for an in replay_angles[pos].unique():
                    mask=replay_angles[pos]==an; rx[mask]=rotate(rx[mask],float(an))
                xb=torch.cat([xb,rx]); yb=torch.cat([yb,y[ri]])
            opt.zero_grad(set_to_none=True); logits=model(xb); loss=F.cross_entropy(logits,yb)
            if a.base=='ewc':
                for old,fisher in ewc:
                    loss=loss+5.*sum((f*(v-o).square()).sum() for v,o,f in zip(model.parameters(),old,fisher))
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.); gn+=float(norm); opt.step(); step+=1
            if any(s.get('t',-1)%50==1 for s in opt.state.values() if 'U' in s): diagnostics.append(flag_snapshot(opt,root/'flags',step,prev))
            loss_sum+=float(loss)*len(idx); correct+=int((logits[:len(idx)].argmax(1)==y[idx]).sum()); total+=len(idx)
        val=[]; test=[]
        for k in ([task] if 'shift' in a.task else range(task+1)):
            va=torch.arange(len(xv),device=device); te=torch.arange(len(xt),device=device)
            if a.task=='split_cifar100': va=va[(yv>=k*10)&(yv<(k+1)*10)]; te=te[(yt>=k*10)&(yt<(k+1)*10)]
            if a.task=='cifar100_class_shift': va=va[(yv>=k*25)&(yv<(k+1)*25)]; te=te[(yt>=k*25)&(yt<(k+1)*25)]
            if not len(va) or not len(te): continue
            ang=k*10 if cls==MNIST else 0
            vw=torch.where(yv[va]%4==k,1.,.1) if a.task=='cifar10_label_shift' else None
            tw=torch.where(yt[te]%4==k,1.,.1) if a.task=='cifar10_label_shift' else None
            val.append(evaluate(model,xv[va],yv[va],ang,vw)); test.append(evaluate(model,xt[te],yt[te],ang,tw))
        val_acc=float(np.mean([z['acc'] for z in val])); test_acc=float(np.mean([z['acc'] for z in test])); seen_best[:len(test)]=np.maximum(seen_best[:len(test)],[z['acc'] for z in test]); forgetting=float(np.mean(seen_best[:len(test)]-[z['acc'] for z in test]))
        row=dict(epoch=ep,task=task,train_loss=loss_sum/total,train_acc=correct/total,val_acc=val_acc,test_acc=test_acc,val_per_task=val,test_per_task=test,forgetting=forgetting,lr=a.lr,gradient_norm=gn/max(1,math.ceil(len(ids)/a.batch)),wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated(),flag_diagnostics=diagnostics)
        with (root/'log.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        last=(last+[row])[-100:]
        if val_acc>best: best=val_acc; best_epoch=ep; torch.save(dict(model=model.state_dict(),epoch=ep,val_acc=val_acc),root/'checkpoints'/'best_val.pt')
        boundary=ep*n_tasks//a.epochs!=((ep-1)*n_tasks//a.epochs)
        if boundary:
            if a.base=='replay':
                replay_ids=torch.cat([replay_ids,ids[:200]]); replay_angles=torch.cat([replay_angles,torch.full((len(ids[:200]),),float(angle),device=device)])
            if a.base=='ewc':
                model.eval(); opt.zero_grad(set_to_none=True); F.cross_entropy(model(rotate(x[ids[:512]],angle)),y[ids[:512]]).backward(); ewc.append(([v.detach().clone() for v in model.parameters()],[v.grad.detach().square() if v.grad is not None else torch.zeros_like(v) for v in model.parameters()]))
        if ep%10==0 or ep in [1,2,5,20,50,100,200,500,1000] or ep==a.epochs:
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep,step=step,best=best,seen_best=seen_best,replay_ids=replay_ids,replay_angles=replay_angles,ewc=ewc)
            torch.save(ck,root/'checkpoints'/f'epoch_{ep:04d}.pt'); torch.save(ck,resume)
        print(json.dumps({k:row[k] for k in ['epoch','val_acc','test_acc','wall_seconds']}),flush=True)
    (root/'summary.json').write_text(json.dumps(dict(complete=True,epochs=a.epochs,best_val=best,best_val_epoch=best_epoch,final=row,last100_val=float(np.mean([z['val_acc'] for z in last])),last100_test=float(np.mean([z['test_acc'] for z in last]))),indent=2))
if __name__=='__main__': main()
