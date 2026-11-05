import argparse,gc,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
import torchvision.transforms as T
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_image_benchmarks import model_for,build_optimizer
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);results=[]
for name in ['resnet18','resnet50','vit_s16','convnext_t']:
 for method in ['adamw','single','strata','galore','independent','sgdm','muon']:
  model=model_for(name,200,64).cuda();cfg=SimpleNamespace(method=method,rank=64,lr=.01 if method=='muon' else .03 if method=='sgdm' else .0003,wd=.0001);opt=build_optimizer(model,cfg)
  x=torch.randn(2,3,64,64,device='cuda');y=torch.tensor([1,2],device='cuda');opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model(x),y)
  loss.backward();opt.step();assert all(torch.isfinite(v).all() for v in model.parameters());results.append(dict(model=name,method=method,finite=True));print(name,method,'passed',flush=True)
  del model,opt,x,y,loss;gc.collect();torch.cuda.empty_cache()
Path(a.out).write_text(json.dumps(results,indent=2))
