"""Validate identities against the actual optimizer; controlled bias diagnostics."""
import hashlib,itertools,json,math,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from strata_pkg.strata_adamw import STRATAAdamW
from strata_pkg import geometry as geo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=root/'analysis/revision_20261101/breadth';out.mkdir(exist_ok=True);rng=np.random.default_rng(42);torch.manual_seed(42);torch.set_num_threads(1)
p=torch.nn.Parameter(torch.randn(8,6,dtype=torch.float64)*.05);opt=STRATAAdamW([p],lr=.001,ranks=[1,2,3],rhos=[.99,.9,.5],beta1s=[.6,.8,.9],beta2s=[.7,.8,.9],a_ks=[1.,.7,.4],a_perp=.2,xi=.02,tau=5,eps=.01,weight_decay=0.);group=opt.param_groups[0];state=opt.state[p];opt._init_matrix_state(state,p,group);original=opt._flag_update;snap={}
def intercept(st,g,gr):
 U=st['U'];cu=[geo.nystrom_apply(y,o,U,gr['gamma']) for y,o in zip(st['Y'],st['Omega'])];H=torch.cat([2*sum(gr['omegas'][k]*cu[k][:,sl] for k in range(j,len(cu))) for j,sl in enumerate(st['slices'])],1);z=float((gr['xi']*geo.horizontal_gradient(U,H,st['slices'])).norm())
 snap.update(z=z,M=[v.clone() for v in st['M']],V=[v.clone() for v in st['V']],cM=list(st['c_M']),cV=list(st['c_V']));original(st,g,gr)
opt._flag_update=intercept
budgets=[];previous_bundle=None;previous_refresh=None;max_gradient=0.
errors={'ambient_momentum':[],'exact_direction':[],'virtual_iterate':[]};w=p.detach().clone();mprev=torch.zeros_like(w);Cprev=lambda x:torch.zeros_like(x)
def maps(us,den,theta,a,cp):
 us=[u.clone() for u in us];den=[s.clone() for s in den];theta=list(theta);a=list(a);proj=sum(u@u.T for u in us);perp=torch.eye(proj.shape[0],dtype=proj.dtype)-proj
 def B(x):return sum(ak*((x@u)/s)@u.T for u,s,ak in zip(us,den,a))+.2*cp*(x@perp)
 def C(x):return .001*sum(ak*th/(1-th)*((x@u)/s)@u.T for u,s,ak,th in zip(us,den,a,theta))
 def BI(x):return sum(ak/(1-th)*((x@u)/s)@u.T for u,s,ak,th in zip(us,den,a,theta))+.2*cp*(x@perp)
 def theta_op(x):return sum(th*(x@u)@u.T for u,th in zip(us,theta))
 return B,C,BI,theta_op
with torch.no_grad():
 for t in range(1,101):
  us=[state['U'][:,s].clone() for s in state['slices']];oldM=[v.clone() for v in state['M']];ages=list(state['c_M']);g=w@torch.diag(torch.arange(1,7,dtype=w.dtype))+torch.tensor(rng.choice([-1.,1.],size=w.shape)*.02);snap.clear();D=opt._matrix_update(state,g,group)
  values=snap if snap else dict(M=state['M'],V=state['V'],cM=state['c_M'],cV=state['c_V']);mh=[m/(1-b**age) for m,b,age in zip(values['M'],group['beta1s'],values['cM'])];vh=[v/(1-b**age) for v,b,age in zip(values['V'],group['beta2s'],values['cV'])];theta=[b*(1-b**(t-1))/(1-b**t) for b in group['beta1s']];perp=torch.eye(6,dtype=w.dtype)-sum(u@u.T for u in us);m=sum(v@u.T for v,u in zip(mh,us))+g@perp
  uerr=sum(th*((om/(1-b**age) if age else torch.zeros_like(om))@u.T-(mprev@u)@u.T) for th,om,b,age,u in zip(theta,oldM,group['beta1s'],ages,us));cp=1/(float((g@perp).norm())/math.sqrt(g.numel())+.01);B,C,BI,TH=maps(us,[v.sqrt()+.01 for v in vh],theta,group['a_ks'],cp)
  errors['ambient_momentum'].append(float((m-TH(mprev)-(g-TH(g))-uerr).norm()));errors['exact_direction'].append(float((D-B(m)).norm()));eta=.001;wnext=w-eta*D;zt=w-Cprev(mprev);rem=Cprev(mprev)-C(mprev)-eta*BI(uerr);znext=wnext-C(m);errors['virtual_iterate'].append(float((znext-(zt-eta*B(g)+rem)).norm()))
  max_gradient=max(max_gradient,float(g.norm()));den=[v.sqrt()+.01 for v in vh];BM=torch.block_diag(*[sum(ak*u@torch.diag(1/s[i])@u.T for u,s,ak in zip(us,den,group['a_ks']))+.2*cp*perp for i in range(8)]);THright=sum(th*u@u.T for th,u in zip(theta,us));entry=dict(step=t,u=float(uerr.norm()),refresh=previous_refresh)
  if previous_bundle is not None:
   pu,pd,pc,pB,pTH,pth=previous_bundle
   bbound=sum(ak*(.01**-2*float((sd-ps).abs().max())+2/.01*float(torch.linalg.matrix_norm(un-up,2))) for ak,sd,ps,un,up in zip(group['a_ks'],den,pd,us,pu))+.2*abs(cp-pc)+.2/.01*float(torch.linalg.matrix_norm(sum(un@un.T-up@up.T for un,up in zip(us,pu)),2))
   tbound=sum(abs(th-pr) for th,pr in zip(theta,pth))+.9*sum(float(torch.linalg.matrix_norm(un@un.T-up@up.T,2)) for un,up in zip(us,pu));entry.update(B_change=float(torch.linalg.matrix_norm(BM-pB,2)),B_bound=bbound,theta_change=float(torch.linalg.matrix_norm(THright-pTH,2)),theta_bound=tbound)
  budgets.append(entry);previous_bundle=([u.clone() for u in us],[v.clone() for v in den],cp,BM,THright,list(theta));previous_refresh=snap.get('z');w=wnext;mprev=m;Cprev=C
assert all(max(v)<1e-10 for v in errors.values()),errors
transport_slacks=[]
for entry in budgets:
 z=entry['refresh']
 if z is None:bound=0.
 else:
  e=(1+z*z)*z*z;assert e<1;bound=.9*math.sqrt(3)*(max_gradient*(1+2/(2-e))*e+2*math.sqrt(4)*max_gradient*(z+e+2*e/(2-e)))
 entry['u_bound']=bound;transport_slacks.append(bound-entry['u'])
bs=[e for e in budgets if 'B_change' in e];assert min(e['B_bound']-e['B_change'] for e in bs)>-1e-10;assert min(e['theta_bound']-e['theta_change'] for e in bs)>-1e-10;assert min(transport_slacks)>-1e-10
(out/'actual_state_budget_trace.json').write_text(json.dumps(budgets,indent=2));fig,axes=plt.subplots(1,2,figsize=(6,2.8),layout='constrained')
axes[0].semilogy([e['step'] for e in bs],[e['B_change'] for e in bs],label='Actual change',color='#0072B2');axes[0].semilogy([e['step'] for e in bs],[e['B_bound'] for e in bs],label='State-derived upper bound',color='#D55E00');axes[0].set(xlabel='Optimizer update',ylabel='Preconditioner variation')
axes[1].semilogy([e['step'] for e in budgets],[e['u'] if e['u']>1e-14 else 1e-18 for e in budgets],label='Actual defect',color='#0072B2');axes[1].semilogy([e['step'] for e in budgets],[e['u_bound'] if e['u_bound']>1e-14 else 1e-18 for e in budgets],label='QR/transport upper bound',color='#D55E00');axes[1].set(xlabel='Optimizer update',ylabel='Transport defect norm')
for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=6)
fig.savefig(root/'paper/figures/fig_actual_state_budgets.pdf',bbox_inches='tight');fig.savefig(root/'paper/figures/fig_actual_state_budgets.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Exact zero-mean asymmetric-noise root and a fixed-seed stochastic approximation.
sigma=.3;eps=.01;v=8*sigma/3+5*eps/4;pred=(math.sqrt(v*v+4*sigma*sigma)-v)/2
f=lambda x:x/(np.abs(x)+eps)
field=lambda x:.1*f(x+3*sigma)+.9*f(x-sigma/3)
assert abs(field(pred))<1e-12 and 0<pred<sigma/3
T=30000;noise_u=rng.random(T);skew=np.where(noise_u<.1,3*sigma,-sigma/3);symmetric=np.where(noise_u<.5,sigma,-sigma);traces={}
for name,noise,normalized in [('Normalized / skew',skew,True),('Normalized / symmetric',symmetric,True),('SGD / skew',skew,False),('SGD / symmetric',symmetric,False)]:
 x=0.;ys=[]
 for t,z in enumerate(noise):
  eta=.3/(t+100)**.65;g=x+z;x-=eta*(f(g) if normalized else g);ys.append(x)
 traces[name]=ys
np.savez_compressed(out/'normalization_bias_traces.npz',**{k:np.array(v) for k,v in traces.items()});plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig=plt.figure(figsize=(6,5.3),layout='constrained');grid=fig.add_gridspec(2,2);ax=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[1,:]),fig.add_subplot(grid[0,1])];xx=np.linspace(-.2,.2,501);ax[0].plot(xx,field(xx),label='Skew normalization',color='#D55E00');ax[0].plot(xx,.5*f(xx+sigma)+.5*f(xx-sigma),label='Symmetric normalization',color='#0072B2');ax[0].axhline(0,color='gray',lw=.7);ax[0].axvline(pred,color='#D55E00',ls=':',lw=1);ax[0].set(xlabel='True gradient h = w',ylabel='Expected normalized update',title='(a) Exact update field');ax[0].legend(frameon=False,fontsize=6)
for name,ys in traces.items():
 z=np.array(ys);smooth=np.convolve(z,np.ones(500)/500,mode='valid');ax[1].plot(np.arange(500,T+1),smooth,label=name,lw=.9,color={'Normalized / skew':'#D55E00','Normalized / symmetric':'#0072B2','SGD / skew':'#009E73','SGD / symmetric':'#333333'}[name])
ax[1].axhline(pred,color='#D55E00',ls='--',lw=.9);ax[1].axhline(0,color='gray',lw=.7);ax[1].set(xlabel='Iteration',ylabel='500-step mean of w',title='(c) Fixed-seed stochastic approximation');ax[1].legend(frameon=False,fontsize=6)
ee=np.logspace(-4,1,200);a=8*sigma/3+5*ee/4;roots=2*sigma*sigma/(np.sqrt(a*a+4*sigma*sigma)+a);ax[2].semilogx(ee,roots,color='#D55E00');ax[2].axhline(sigma/3,color='gray',ls=':');ax[2].set(xlabel='Epsilon',ylabel='Biased stationary point',title='(b) Closed-form stationary bias')
for a in ax:a.grid(alpha=.2)
fig.savefig(root/'paper/figures/fig_actual_normalization_theory.pdf');fig.savefig(root/'paper/figures/fig_actual_normalization_theory.png',dpi=200);plt.close(fig)
# Optimized posterior variance: derivative zero and refinement Jensen gap.
variance_errors=[];gaps=[]
for _ in range(100):
 dims=np.array([3,5,8]);h=rng.uniform(0,2,3);lam=20.;s0=.3;s=s0/(1+2*lam*s0*h);derivative=dims*h+dims/(2*lam)*(1/s0-1/s);variance_errors.append(max(abs(derivative)));fine=sum(dims/(2*lam)*np.log1p(2*lam*s0*h));coarse=sum(dims)/(2*lam)*np.log1p(2*lam*s0*np.average(h,weights=dims));gaps.append(coarse-fine)
assert max(variance_errors)<1e-10 and min(gaps)>=-1e-12
summary=dict(minimum_preconditioner_bound_slack=min(e['B_bound']-e['B_change'] for e in bs),minimum_theta_bound_slack=min(e['theta_bound']-e['theta_change'] for e in bs),minimum_transport_bound_slack=min(transport_slacks),actual_implementation_identity_errors={k:max(v) for k,v in errors.items()},steps=100,flag_refreshes=20,asymmetric_noise_root=pred,root_residual=float(abs(field(pred))),tail_mean={k:float(np.mean(v[-5000:])) for k,v in traces.items()},posterior_derivative_error=max(variance_errors),minimum_refinement_gain=min(gaps),optimizer_sha256=hashlib.sha256((root/'experiments/strata_pkg/strata_adamw.py').read_bytes()).hexdigest(),synthetic_seed=42,all_identity_checks_passed=True)
(out/'actual_theory_validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
