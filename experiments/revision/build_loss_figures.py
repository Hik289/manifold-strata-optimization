"""Full raw trajectories, frozen selection rules, and source-level plot audit."""
import collections,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_table_style import loss_style
root=Path(__file__).resolve().parents[2];figdir=root/'paper/figures';report=root/'analysis/revision_20261101/breadth';report.mkdir(exist_ok=True);audit=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight','grid.alpha':.18})
loss_style(plt)
colors={'strata':'#D55E00','single':'#0072B2','adamw':'#333333','galore':'#009E73','independent':'#CC79A7'};names={'strata':'STRATA','single':'Single-Grassmann','adamw':'AdamW','galore':'GaLore','independent':'Independent'}
def read(p):
 rows=[]
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 return rows
def line(ax,p,method,key,xkey='epoch',label=None):
 rows=read(p);x=[];y=[]
 for r in rows:
  if key=='seen_val_loss':v=np.mean([z['loss'] for z in r['val_per_task']])
  else:v=r.get(key)
  if v is not None:x.append(r[xkey]);y.append(float(v))
 yy=np.array(y);xx=np.array(x);yy[~np.isfinite(yy)]=np.nan
 if len(x):
  ax.plot(xx,yy,lw=1.15,alpha=.88,color=colors[method],label=label or names[method]);ax.set_xlim(0,max(x));ax.grid(True)
  if xkey=='epoch' and len(x)>=1000:
   endpoint=1000*math.ceil(max(x)/1000);ax.set_xlim(0,endpoint);ax.set_xticks(np.linspace(0,endpoint,5 if endpoint==1000 else 7))
 audit.append(dict(figure=current,source=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),method=method,metric=key,points=len(x),last_x=max(x) if x else None,smoothing='none'))
def finish(fig,name):
 legend={}
 for axis in fig.axes:
  hs,ls=axis.get_legend_handles_labels()
  for h,l in zip(hs,ls):legend.setdefault(l,h)
 if legend:fig.legend(list(legend.values()),list(legend),loc='upper center',bbox_to_anchor=(.5,1.045),ncol=5,frameon=False)
 fig.savefig(figdir/(name+'.pdf'));fig.savefig(figdir/(name+'.png'),dpi=200);plt.close(fig)
# Classification: complete trials only, rank64, terminal-100 validation accuracy.
current='fig_loss_classification_validation';fig,axes0=plt.subplots(4,2,figsize=(6,6.4),layout='constrained');axes=axes0.T;selection=[]
for col,(ds,folder) in enumerate([('mnist','runs3_grid'),('fashion','runs3_grid'),('cifar10','runs3_cnn'),('cifar100','runs3_cnn')]):
 for method in ['strata','single','adamw','galore']:
  raw='single_grassmann' if method=='single' else method;candidates=list((root/'experiments'/folder).glob(f'{ds}__{raw}__*_log.jsonl'))
  if method in ['adamw','galore'] and ds.startswith('cifar'):candidates += list((root/'experiments/runs3').glob(f'{ds}_{raw}_log.jsonl'))
  candidates=[p for p in candidates if len(read(p))>=1000 and all(math.isfinite(r.get('val_acc',float('nan'))) for r in read(p)[-100:])]
  if not candidates:continue
  best=max(candidates,key=lambda p:np.mean([r['val_acc'] for r in read(p)[-100:]]));selection.append(dict(dataset=ds,method=method,source=str(best.relative_to(root)),rule='terminal100 validation accuracy, complete 1000epoch trials only',trials=len(candidates)))
  line(axes[0,col],best,method,'train_loss');line(axes[1,col],best,method,'val_loss')
 axes[0,col].set_title({'mnist':'MNIST / MLP','fashion':'Fashion-MNIST / MLP','cifar10':'CIFAR-10 / ResNet-18','cifar100':'CIFAR-100 / ResNet-18'}[ds])
 for row in [0,1]:axes[row,col].set_yscale('log');axes[row,col].set_xlabel('Epoch')
axes[0,0].set_ylabel('Training cross-entropy');axes[1,0].set_ylabel('Validation cross-entropy');finish(fig,current)
# Compact main-text loss plot; full train/validation curves stay in appendix.
current='fig_loss_classification';fig,axs=plt.subplots(2,2,figsize=(6,4.5),layout='constrained')
for ax,ds in zip(axs.flat,['mnist','fashion','cifar10','cifar100']):
 for item in selection:
  if item['dataset']==ds:line(ax,root/item['source'],item['method'],'train_loss')
 ax.set_title({'mnist':'MNIST / MLP','fashion':'Fashion-MNIST / MLP','cifar10':'CIFAR-10 / ResNet-18','cifar100':'CIFAR-100 / ResNet-18'}[ds]);ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Training cross-entropy')
finish(fig,current)
# Strict rank64 PDE comparison, distinct from the old pooled-rank grid optimum.
current='fig_loss_scientific';fig,axes0=plt.subplots(3,2,figsize=(6,6.0),layout='constrained');axes=axes0.T;pinn_selected={}
for col,ds in enumerate(['burgers','burgersg2','poisson']):
 pinn_selected[ds]={}
 for method,raw in [('strata','strata'),('single','single_grassmann')]:
  candidates=list((root/'experiments/runs2_grid_eps1e6').glob(f'{ds}__{raw}__*__R64/log.jsonl'))
  candidates=[p for p in candidates if len(read(p))>=3000 and all(math.isfinite(r['loss_pde']) for r in read(p)[-100:])]
  best=min(candidates,key=lambda p:np.mean([r['loss_pde'] for r in read(p)[-100:]]));pinn_selected[ds][method]=dict(source=str(best.relative_to(root)),pde=np.mean([r['loss_pde'] for r in read(best)[-100:]]),rel_l2=np.mean([r['rel_l2'] for r in read(best)[-100:]]))
  line(axes[0,col],best,method,'train_loss');line(axes[1,col],best,method,'loss_pde')
 axes[0,col].set_title({'burgers':'Burgers','burgersg2':'Burgers-g2','poisson':'Poisson'}[ds])
 for row in [0,1]:axes[row,col].set_yscale('log');axes[row,col].set_xlabel('Epoch')
axes[0,0].set_ylabel('Total training loss');axes[1,0].set_ylabel('PDE residual loss');finish(fig,current)
# Old graph plus the newly completed graph and continual-learning cohorts.
live=report/'live';jobs=json.loads((live/'queue_snapshot.json').read_text());complete=[j for j in jobs if j['status']=='complete'];current='fig_loss_graph_continual';fig,axes0=plt.subplots(3,2,figsize=(6,6.0),layout='constrained');axes=axes0.T
for method in ['strata','single','adamw','galore']:
 p=root/'experiments/runs2_graph'/f'{"single_grassmann" if method=="single" else method}_log.jsonl'
 if p.exists():line(axes[0,0],p,method,'train_loss');line(axes[1,0],p,method,'val_loss')
for col,prefix in [(1,'Cora__gcn__'),(2,'rotated_mnist_cl__finetune__')]:
 for method in colors:
  js=[j for j in complete if Path(j['out']).name.startswith(prefix+method+'__') and '__lr0.0003' in j['out']]
  if not js:continue
  p=live/js[0]['out']/'log.jsonl';line(axes[0,col],p,method,'train_loss');line(axes[1,col],p,method,'val_loss' if col==1 else 'seen_val_loss')
for col,title in enumerate(['ogbn-arxiv / GraphSAGE (archive)','Cora / GCN (new, lr=0.0003)','Rotated-MNIST / CL (new, lr=0.0003)']):
 axes[0,col].set_title(title)
 for row in [0,1]:axes[row,col].set_yscale('log');axes[row,col].set_xlabel('Epoch')
axes[0,0].set_ylabel('Training cross-entropy');axes[1,0].set_ylabel('Validation cross-entropy');finish(fig,current)
# Ongoing traces are separated and never promoted to full-budget endpoints.
current='fig_loss_ongoing';fig,axes0=plt.subplots(4,2,figsize=(6,6.4),layout='constrained');axes=axes0.T
for col,prefix in enumerate(['wikitext103__gpt2-small__','tinyimagenet__resnet18__finetune__','imagenet100__resnet18__finetune__','wikitext2_words__lstm__']):
 for method in ['strata','single']:
  js=[j for j in jobs if Path(j['out']).name.startswith(prefix+method+'__') and '__lr0.0003' in j['out']]
  for j in js[:1]:
   p=live/j['out']/'log.jsonl'
   if not p.exists():continue
   xkey='tokens' if col==0 else 'epoch'
   line(axes[0,col],p,method,'train_loss',xkey);line(axes[1,col],p,method,'val_loss',xkey)
 titles=['WikiText-103 / GPT-2','Tiny-ImageNet / R18','ImageNet-100 / R18','WikiText-2 / LSTM'];completed=any(Path(j['out']).name.startswith(prefix) and '__strata__' in j['out'] and '__lr0.0003' in j['out'] and j['status']=='complete' for j in jobs);axes[0,col].set_title(titles[col]+(' (STRATA complete)' if completed else ' (ongoing)'))
 for row in [0,1]:axes[row,col].set_xlabel('Training tokens' if col==0 else 'Epoch')
axes[0,0].set_ylabel('Training cross-entropy');axes[1,0].set_ylabel('Validation cross-entropy');finish(fig,current)
# Full-grid losses, including adverse outcomes, split into readable task figures.
for ds,folder in [('mnist','runs3_grid'),('fashion','runs3_grid'),('cifar10','runs3_cnn'),('cifar100','runs3_cnn')]:
 current='fig_loss_grid_'+ds;fig,axs=plt.subplots(2,2,figsize=(6,4.5),layout='constrained')
 for ax,(method,raw) in zip(axs.flat,[('strata','strata'),('single','single_grassmann'),('adamw','adamw'),('galore','galore')]):
  ps=list((root/'experiments'/folder).glob(f'{ds}__{raw}__*_log.jsonl'))
  if ds.startswith('cifar') and method in ['adamw','galore']:ps+=list((root/'experiments/runs3').glob(f'{ds}_{raw}_log.jsonl'))
  for k,p in enumerate(sorted(ps)):
   rows=read(p)
   if len(rows)<1000:continue
   vals=np.array([r['train_loss'] for r in rows]);label=p.name.split('__')[-2] if '__' in p.name else 'archive'
   ax.plot([r['epoch'] for r in rows],vals,lw=.9,alpha=.8,label=label,color=plt.cm.viridis(.15+.7*k/max(1,len(ps)-1)))
  ax.set_title(ds+' / '+names[method]);ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Training cross-entropy');ax.grid(True)
  if len(ax.lines):ax.legend(frameon=False,fontsize=6)
  else:ax.set_axis_off();ax.text(.5,.5,'No completed archived trial',ha='center',va='center',transform=ax.transAxes,color='gray',fontsize=8)
 fig.savefig(figdir/(current+'.pdf'));fig.savefig(figdir/(current+'.png'),dpi=180);plt.close(fig)
# Loss AUC/gradient summaries using observed full horizons; no interpolation.
summary=[]
for item in selection:
 p=root/item['source'];rows=read(p);loss=np.array([r['train_loss'] for r in rows]);xx=np.array([r['epoch'] for r in rows]);norms=[r.get('grad_norm',r.get('gradient_norm')) for r in rows[-100:]]
 norm_status='recorded' if all(v is not None and math.isfinite(v) for v in norms) else 'not_logged' if all(v is None for v in norms) else 'incomplete_or_nonfinite'
 gn=float(np.mean(norms)) if norm_status=='recorded' else None
 summary.append(item|dict(epochs=len(rows),loss_auc=float(np.trapz(loss,xx)),terminal100_train_loss=float(np.mean(loss[-100:])),terminal100_gradient_norm=gn,gradient_norm_status=norm_status))
(report/'loss_curve_audit.json').write_text(json.dumps(dict(curves=audit,classification_selection=selection,pinn_rank64_selection=pinn_selected,summaries=summary),indent=2))
with (report/'convergence_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print(json.dumps(dict(curves=len(audit),classification_selected=len(selection),pinn_rank64=pinn_selected,figures=9),indent=2))
