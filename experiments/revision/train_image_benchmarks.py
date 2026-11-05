"""Full-data large image/continual benchmarks backed by validated manifests.
No pretrained weights, no reduced-data substitutes, fixed seed42 and 1000 epochs.
"""
import argparse,hashlib,io,json,math,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18,resnet50,convnext_tiny
from torchvision.models.vision_transformer import VisionTransformer
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stream import optimizer as compressed_optimizer,flag_snapshot,tensor_state_bytes

class Records(Dataset):
 def __init__(self,rows,transform):self.rows=rows;self.transform=transform;self.arrow_cache={}
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  if r.get('storage')=='hf_arrow':
   from datasets import Dataset as HFDataset,concatenate_datasets,Image as HFImage
   cache,ref=r['path'].split('#');official,index=ref.split('/');key=(cache,official)
   if key not in self.arrow_cache:self.arrow_cache[key]=concatenate_datasets([HFDataset.from_file(str(f)) for f in sorted(Path(cache).glob(f'imagenet-100-{official}*.arrow'))]).cast_column('image',HFImage(decode=False))
   picture=self.arrow_cache[key][int(index)]['image'];im=Image.open(io.BytesIO(picture['bytes'])) if picture.get('bytes') is not None else Image.open(picture['path'])
  else:im=Image.open(r['path'])
  with im:x=self.transform(im.convert('RGB'))
  return x,int(r['label'])
def model_for(name,n,size):
 if name=='resnet18':return resnet18(weights=None,num_classes=n)
 if name=='resnet50':return resnet50(weights=None,num_classes=n)
 if name=='convnext_t':return convnext_tiny(weights=None,num_classes=n)
 return VisionTransformer(image_size=size,patch_size=16,num_layers=12,num_heads=6,hidden_dim=384,mlp_dim=1536,num_classes=n)
def build_optimizer(model,a):
 if a.method=='sgdm':return torch.optim.SGD(model.parameters(),lr=a.lr,momentum=.9,weight_decay=a.wd)
 if a.method=='muon':
  from vendor.muon_official import SingleDeviceMuonWithAuxAdam
  hidden=[];aux=[]
  for n,p in model.named_parameters():
   is_aux=p.ndim not in [2,4] or n.startswith(('fc.','heads.','classifier.','conv_proj.','conv1.','features.0.')) or 'pos_embedding' in n or 'class_token' in n
   (aux if is_aux else hidden).append(p)
  return SingleDeviceMuonWithAuxAdam([dict(params=hidden,use_muon=True,lr=a.lr,momentum=.95,weight_decay=a.wd),dict(params=aux,use_muon=False,lr=.03*a.lr,betas=(.9,.999),eps=1e-6,weight_decay=a.wd)])
 opt=compressed_optimizer(model,a)
 for g in opt.param_groups:g['weight_decay']=a.wd
 return opt

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['imagenet100','tinyimagenet','domainnet','core50'],required=True);p.add_argument('--model',choices=['resnet18','resnet50','vit_s16','convnext_t'],default='resnet18');p.add_argument('--method',choices=['adamw','single','strata','galore','independent','sgdm','muon'],required=True);p.add_argument('--base',choices=['finetune','replay','ewc'],default='finetune');p.add_argument('--epochs',type=int,default=1000);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--rank',type=int,default=64);p.add_argument('--wd',type=float,default=.0001);p.add_argument('--batch',type=int,default=64);p.add_argument('--workers',type=int,default=4);p.add_argument('--image-size',type=int);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert a.smoke or a.epochs>=1000
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 dataset=Path(a.data)/a.dataset;ready=json.loads((dataset/'ready.json').read_text());assert ready['ready'];manifest=dataset/'manifest.jsonl';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==ready['manifest_sha256']
 rows=[json.loads(s) for s in manifest.read_text().splitlines()];nt=ready['n_tasks'];n=ready['n_classes'];size=a.image_size or (224 if a.dataset=='imagenet100' else 64 if a.dataset=='tinyimagenet' else 128)
 root=Path(a.out);root.mkdir(parents=True,exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True);(root/'flags').mkdir(exist_ok=True)
 if (root/'summary.json').exists():return
 if a.smoke:
  # Test-only mode: one train/val/test batch from a real manifest, never a formal result.
  rows=[r for split in ['train','validation','test'] for r in [x for x in rows if x['split']==split][:a.batch*2]];nt=1
  for r in rows:r['task']=0
 train_tf=T.Compose([T.RandomResizedCrop(size,scale=(.7,1.)),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]);test_tf=T.Compose([T.Resize(round(size*256/224)),T.CenterCrop(size),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
 groups={split:[[r for r in rows if r['split']==split and r['task']==task] for task in range(nt)] for split in ['train','validation','test']};assert all(groups['train']) and all(groups['validation']) and all(groups['test'])
 model=model_for(a.model,n,size).cuda();opt=build_optimizer(model,a)
 for group in opt.param_groups:group['initial_lr']=group['lr']
 info=vars(a)|dict(seed=42,image_size=size,architecture=a.model,pretrained=False,schedule='cosine with per-task warmup/restart; optimizer state carried across tasks',precision='BF16 autocast; FP32 weights and state',manifest_sha256=ready['manifest_sha256'],data_source=ready['source'],data_protocol=ready['protocol'],n_tasks=nt,n_classes=n,selection='validation only, compare terminal-100 means once full search is complete',replay_capacity=2000,ewc='online sum of task quadratic penalties; diagonal batch-gradient-square Fisher proxy, 8 training batches; lambda10')
 (root/'config.json').write_text(json.dumps(info,indent=2));step=0;start=0;best=-1.;prev={};history=[];buffer=[];precision={};centers={};best_val_by_task=np.zeros(nt);best_test_by_task=np.zeros(nt);phase_end=[];fwt=[]
 @torch.no_grad()
 def evaluate(records):
  assert records
  loader=DataLoader(Records(records,test_tf),batch_size=a.batch,num_workers=a.workers,shuffle=False,pin_memory=True);model.eval();ls=correct=total=0
  for x,y in loader:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x);loss=nn.functional.cross_entropy(z,y,reduction='sum')
   ls+=float(loss);correct+=int((z.argmax(1)==y).sum());total+=len(y)
  return dict(loss=ls/total,acc=correct/total,n=total)
 initial_file=root/'untrained_validation.json'
 if initial_file.exists():initial=json.loads(initial_file.read_text())
 else:
  initial=[evaluate(group) for group in groups['validation']];initial_file.write_text(json.dumps(initial,indent=2))
 resume=root/'checkpoints/latest.pt'
 if resume.exists():
  ck=torch.load(resume,map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch'];step=ck['step'];best=ck['best'];buffer=ck['buffer'];precision=ck['precision'];centers=ck['centers'];best_val_by_task=ck['best_val_by_task'];best_test_by_task=ck['best_test_by_task'];phase_end=ck['phase_end'];fwt=ck['fwt']
  if (root/'log.jsonl').exists():history=[json.loads(s) for s in (root/'log.jsonl').read_text().splitlines()][-100:]
 else:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=0,step=0),root/'checkpoints/epoch_0000.pt')
 def checkpoint(epoch):
  return dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,best=best,buffer=buffer,precision=precision,centers=centers,best_val_by_task=best_val_by_task,best_test_by_task=best_test_by_task,phase_end=phase_end,fwt=fwt)
 for ep in range(start+1,a.epochs+1):
  torch.manual_seed(42+ep);random.seed(42+ep);np.random.seed(42+ep);task=min(nt-1,(ep-1)*nt//a.epochs);t0=time.monotonic();newtask=ep==1 or task!=min(nt-1,(ep-2)*nt//a.epochs)
  if newtask and task>0:
   before=evaluate(groups['validation'][task]);fwt.append(dict(task=task,pre_task_val_acc=before['acc'],initial_val_acc=initial[task]['acc'],forward_transfer=before['acc']-initial[task]['acc']))
  loader=DataLoader(Records(groups['train'][task],train_tf),batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,generator=torch.Generator().manual_seed(42+ep))
  replay_loader=DataLoader(Records(buffer,train_tf),batch_size=max(1,a.batch//4),shuffle=True,num_workers=min(2,a.workers),pin_memory=True,generator=torch.Generator().manual_seed(42000+ep)) if a.base=='replay' and buffer else None;replay_iter=iter(replay_loader) if replay_loader else None
  model.train();total=correct=ls=gradnorm=steps=0;diagnostics=[];torch.cuda.reset_peak_memory_stats();lo=math.ceil(task*a.epochs/nt)+1;hi=math.ceil((task+1)*a.epochs/nt);local_epoch=ep-lo+1;factor=min(1.,local_epoch/min(20,max(1,(hi-lo+1)//5)))*.5*(1+math.cos(math.pi*(local_epoch-1)/max(1,hi-lo+1)))
  for group in opt.param_groups:group['lr']=group['initial_lr']*factor
  for x,y in loader:
   if replay_iter is not None:
    try:rx,ry=next(replay_iter)
    except StopIteration:replay_iter=iter(replay_loader);rx,ry=next(replay_iter)
    x=torch.cat([x,rx]);y=torch.cat([y,ry])
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x);ce=nn.functional.cross_entropy(z,y)
   loss=ce
   if a.base=='ewc' and precision:loss=loss+5.*sum((precision[n]*(v-centers[n]).square()).sum() for n,v in model.named_parameters() if n in precision)
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();step+=1;steps+=1;gradnorm+=float(gn);ls+=float(ce)*len(y);correct+=int((z.argmax(1)==y).sum());total+=len(y)
   if any(s.get('t',-1)%50==1 for s in opt.state.values() if 'U' in s):diagnostics.append(flag_snapshot(opt,root/'flags',step,prev))
  val=[evaluate(r) for r in groups['validation'][:task+1]];test=[evaluate(r) for r in groups['test'][:task+1]];vacc=float(np.mean([r['acc'] for r in val]));tacc=float(np.mean([r['acc'] for r in test]));best_val_by_task[:task+1]=np.maximum(best_val_by_task[:task+1],[r['acc'] for r in val]);best_test_by_task[:task+1]=np.maximum(best_test_by_task[:task+1],[r['acc'] for r in test])
  row=dict(epoch=ep,task=task,train_loss=ls/total,train_acc=correct/total,val_acc=vacc,test_acc=tacc,val_loss=float(np.mean([r['loss'] for r in val])),test_loss=float(np.mean([r['loss'] for r in test])),val_per_task=val,test_per_task=test,forgetting=float(np.mean(best_test_by_task[:task+1]-[r['acc'] for r in test])),val_forward_transfer=float(np.mean([r['forward_transfer'] for r in fwt])) if fwt else 0.,gradient_norm=gradnorm/steps,lr=a.lr*factor,wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated(),optimizer_state_bytes=tensor_state_bytes(opt),flag_diagnostics=diagnostics)
  with (root/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  history=(history+[row])[-100:]
  if vacc>best:best=vacc;torch.save(dict(model=model.state_dict(),epoch=ep,val_acc=vacc),root/'checkpoints/best_val.pt')
  boundary=ep==a.epochs or min(nt-1,ep*nt//a.epochs)!=task
  if boundary:
   phase_end.append(dict(epoch=ep,task=task,val=val,test=test))
   if a.base=='replay':
    # Fixed-size reservoir over all past training examples, keyed independently of outcomes.
    candidates=buffer+groups['train'][task];buffer=sorted(candidates,key=lambda r:hashlib.sha256(('replay42/'+r['path']).encode()).hexdigest())[:2000]
   if a.base=='ewc':
    model.eval();new={n:torch.zeros_like(v) for n,v in model.named_parameters()};fl=DataLoader(Records(groups['train'][task][:a.batch*8],test_tf),batch_size=a.batch,num_workers=a.workers)
    batches=0
    for x,y in fl:
     opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.cross_entropy(model(x.cuda()),y.cuda())
     loss.backward();batches+=1
     for n,v in model.named_parameters():
      if v.grad is not None:new[n].add_(v.grad.detach().square())
    for n,v in model.named_parameters():
     f=new[n]/max(1,batches);old=precision.get(n,torch.zeros_like(f));den=old+f;centers[n]=((old*centers[n] if n in centers else 0)+f*v.detach())/den.clamp_min(1e-12);precision[n]=den
  if ep%10==0 or ep in [1,2,5,20,50,100,200,500,1000] or ep==a.epochs:
   ck=checkpoint(ep);torch.save(ck,root/'checkpoints'/f'epoch_{ep:04d}.pt');torch.save(ck,resume)
  print(json.dumps({k:row[k] for k in ['epoch','task','val_acc','test_acc','wall_seconds']}),flush=True)
 (root/'summary.json').write_text(json.dumps(dict(complete=True,epochs=a.epochs,final=row,best_val=best,last100_val=float(np.mean([r['val_acc'] for r in history])),last100_test=float(np.mean([r['test_acc'] for r in history])),phase_end=phase_end,forward_transfer=fwt),indent=2))
if __name__=='__main__':main()
