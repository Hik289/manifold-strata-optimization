"""Planetoid citation-node benchmarks with fixed public masks and seed 42."""
import argparse,json,sys,time,math,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stream import optimizer,flag_snapshot
class GNN(nn.Module):
    def __init__(self,d,n,kind):
        super().__init__();self.kind=kind;self.first=nn.Linear(d,256);self.layers=nn.ModuleList([nn.Linear(512 if kind=='sage' else 256,256) for _ in range(3)]);self.out=nn.Linear(256,n);self.att=nn.Parameter(torch.zeros(3,2,256)) if kind=='gat' else None
    def forward(self,x,adj,edge):
        h=F.relu(self.first(x));initial=h
        for k,layer in enumerate(self.layers):
            h=F.dropout(h,p=.5,training=self.training)
            if self.kind=='gat':
                src,dst=edge; score=F.leaky_relu((h[src]*self.att[k,0]).sum(1)+(h[dst]*self.att[k,1]).sum(1),.2)
                mx=score.new_full((len(h),),-torch.inf);mx.scatter_reduce_(0,dst,score,reduce='amax',include_self=True);weight=(score-mx[dst]).exp();den=weight.new_zeros(len(h));den.index_add_(0,dst,weight);agg=h.new_zeros(h.shape);agg.index_add_(0,dst,h[src]*(weight/den[dst].clamp_min(1e-12))[:,None])
            else:agg=torch.sparse.mm(adj,h)
            if self.kind=='sage':h=F.relu(layer(torch.cat([h,agg],1)))
            elif self.kind=='gcnii':h=F.relu(.9*layer(.9*agg+.1*initial)+.1*(.9*agg+.1*initial))
            else:h=F.relu(layer(agg))
        return self.out(h)
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['Cora','CiteSeer','PubMed']);p.add_argument('--model',choices=['gcn','gat','sage','gcnii'],default='gcn');p.add_argument('--method',required=True);p.add_argument('--lr',type=float,default=.001);p.add_argument('--rank',type=int,default=64);p.add_argument('--epochs',type=int,default=1000);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--ablation',default='full',choices=['full','same_rho','reverse_rho','equal_a','no_residual','random_flag','no_transport','frozen_after_warmup','heterogeneous_memory','periodic_svd','least_squares_transport']);p.add_argument('--flag-warmup-steps',type=int,default=200);p.add_argument('--flag-xi',type=float);a=p.parse_args();assert a.epochs>=1000 or a.smoke;assert a.flag_xi is None or (a.flag_xi>0 and a.ablation=='full')
    from torch_geometric.datasets import Planetoid
    torch.set_num_threads(4);torch.manual_seed(42);ds=Planetoid(str(Path(a.data)/'raw/Planetoid'),a.dataset);d=ds[0].cuda();x=d.x;x=x/x.sum(1,keepdim=True).clamp_min(1);edge=d.edge_index
    selfloop=torch.arange(len(x),device='cuda');edge=torch.cat([edge,torch.stack([selfloop,selfloop])],1);src,dst=edge;deg=torch.bincount(dst,minlength=len(x)).float();val=1/(deg[src]*deg[dst]).sqrt();adj=torch.sparse_coo_tensor(torch.stack([dst,src]),val,(len(x),len(x))).coalesce()
    model=GNN(x.shape[1],ds.num_classes,a.model).cuda();opt=optimizer(model,a);root=Path(a.out);root.mkdir(parents=True,exist_ok=True);(root/'flags').mkdir(exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True);(root/'config.json').write_text(json.dumps(vars(a)|dict(seed=42,split='Planetoid public fixed masks',normalization='row-normalized features; symmetric adjacency with self loops',ablation=a.ablation if a.flag_xi is None else f'xi_{a.flag_xi:g}',flag_xi_effective=.001 if a.flag_xi is None else a.flag_xi,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),logger_version='float32 bases; float64 QR/SVD span diagnostics; epoch0 checkpoint'),indent=2));best=-1;prev={}
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=0),root/'checkpoints'/'epoch_0000.pt')
    for ep in range(1,a.epochs+1):
        t0=time.monotonic();model.train();opt.zero_grad(set_to_none=True);z=model(x,adj,edge);loss=F.cross_entropy(z[d.train_mask],d.y[d.train_mask]);
        if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite graph loss at epoch {ep}')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();diag=flag_snapshot(opt,root/'flags',ep,prev) if ep%50==1 else {}
        model.eval()
        with torch.no_grad():
            z=model(x,adj,edge);metrics={}
            for name,mask in [('train',d.train_mask),('val',d.val_mask),('test',d.test_mask)]:metrics[name+'_acc']=float((z[mask].argmax(1)==d.y[mask]).float().mean());metrics[name+'_loss']=float(F.cross_entropy(z[mask],d.y[mask]))
        row=dict(epoch=ep,**metrics,gradient_norm=float(gn),lr=a.lr,wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated(),flag_diagnostics=diag)
        with (root/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if row['val_acc']>best:best=row['val_acc'];torch.save(dict(model=model.state_dict(),epoch=ep),root/'checkpoints'/'best_val.pt')
        if ep%10==0 or ep in [1,2,5,20,50,100,200,500,1000]:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep),root/'checkpoints'/f'epoch_{ep:04d}.pt')
        if ep%20==0:print(json.dumps(row),flush=True)
    (root/'summary.json').write_text(json.dumps(dict(complete=True,best_val=best,final=row),indent=2))
if __name__=='__main__':main()
