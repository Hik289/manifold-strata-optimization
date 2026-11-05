"""Independent finite-dimensional checks and equality witnesses, not proofs."""
from pathlib import Path
import json, math
import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.special import gammaln, xlogy

rng = np.random.default_rng(42)
out = {}

def orth(d):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q @ np.diag(np.sign(np.diag(r)))

def blocks(q, sizes):
    cuts = np.cumsum([0, *sizes])
    return [q[:, a:b] @ q[:, a:b].T for a, b in zip(cuts[:-1], cuts[1:])]

def metric_check(p, v):
    k = len(p) - 1
    t = np.array([[np.trace(a @ b) for b in v] for a in p])
    dp = sum(np.linalg.norm(sum(p[:i])-sum(v[:i]), 'fro')**2 for i in range(1,k+1))
    dpi = sum(np.linalg.norm(a-b, 'fro')**2 for a,b in zip(p,v))
    flow = sum(abs(j-a)*t[j,a] for j in range(k+1) for a in range(k+1))
    assert abs(dp-flow) < 1e-10
    assert dpi <= 2*dp+1e-10 and dp <= k*dpi/2+1e-10
    partial = dpi-np.linalg.norm(p[-1]-v[-1], 'fro')**2
    assert dp <= k*partial+1e-10
    return abs(dp-flow),dp,dpi

errs=[]
for k in range(1,7):
    for _ in range(50):
        sizes=rng.integers(1,4,size=k+1);d=sum(sizes)
        errs.append(metric_check(blocks(orth(d),sizes),blocks(orth(d),sizes))[0])
witnesses=[]
for k in range(1,7):
    d=k+1;p=blocks(np.eye(d),[1]*d)
    for j in [1,k]:
        theta=.37;q=np.eye(d);q[np.ix_([0,j],[0,j])]=[[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
        _,dp,dpi=metric_check(p,blocks(q,[1]*d));witnesses.append([k,j,dp/dpi])
        assert abs(dp/dpi-j/2)<1e-12
out['overlap_transport']={'random_cases':300,'max_identity_error':max(errs),'equality_witnesses_K_distance_ratio':witnesses}

dk=[]
for d in [3,8,15]:
    for rank in range(1,d):
        lam=np.arange(d,0,-1,dtype=float);c=np.diag(lam)
        for _ in range(10):
            e=rng.normal(size=(d,d));e=(e+e.T)/2;e*=.2/np.linalg.norm(e,2)
            val,q=np.linalg.eigh(c+e);idx=np.argsort(val)[::-1];val=val[idx];q=q[:,idx];qh=q[:,:rank]
            gap=val[rank-1]-lam[rank]
            residual=np.linalg.norm(e[rank:,:]@qh,'fro')
            lhs=np.linalg.norm(qh@qh.T-np.diag([1]*rank+[0]*(d-rank)),'fro')
            bound=math.sqrt(2)*residual/gap
            assert lhs<=bound+1e-10
            assert bound<=math.sqrt(2*min(rank,d-rank))*.2/.8+1e-10
            dk.append(bound-lhs)
out['davis_kahan']={'cases':len(dk),'minimum_residual_bound_slack':min(dk)}

freedman=[]
for _ in range(1000):
    v,r,u=10**rng.uniform(-8,5,size=3);a=r*u/3;rad=a+math.sqrt(2*v*u+a*a)
    eq=abs(rad*rad-2*u*(v+r*rad/3))/max(1,rad*rad)
    assert eq<1e-12 and rad<=math.sqrt(2*v*u)+2*r*u/3+1e-9*max(1,rad)
    freedman.append(eq)
out['freedman_quadratic_inversion']={'cases':1000,'max_relative_residual':max(freedman)}

for _ in range(1000):
    mu=10**rng.uniform(-2,1);L=mu*10**rng.uniform(0,2);eta=2/(L+mu)
    q=(L-mu)/(L+mu)
    assert abs(max(abs(1-eta*mu),abs(1-eta*L))-q)<1e-12
    b,sig=rng.uniform(0,2,size=2);x=(q*b+math.sqrt(b*b+(1-q*q)*eta*eta*sig*sig))/(1-q*q)
    assert abs(x*x-((q*x+b)**2+eta*eta*sig*sig))<1e-8*max(1,x*x)
out['tracking']={'cases':1000,'optimal_step_and_mixed_fixed_point_checked':True,'pure_drift_gain':'1/(1-q)','pure_noise_variance_gain':'eta^2/(1-q^2)'}

slacks=[];equality=[]
for _ in range(300):
    d=5;L=2.;ev=10**rng.uniform(-3,1,size=d);B=orth(d);B=B@np.diag(ev)@B.T;m=min(ev);M=max(ev)
    hess=orth(d);hess=hess@np.diag(rng.uniform(0,L,d))@hess.T
    w,z,ec,er=rng.normal(size=(4,d));h=hess@w;eta=1/(4*L*M);loss=lambda x:float(x@hess@x/2)
    vals=[];eps2=[];epsdual=[]
    for sign in [-1,1]:
        eps=ec+sign*er;vals.append(loss(w-eta*(B@(h+sign*z)+eps)));eps2.append(eps@eps);epsdual.append(eps@np.linalg.solve(B,eps))
    nu=float(np.linalg.norm(B@z)**2)
    upper=loss(w)-m*eta*np.linalg.norm(h)**2/2+L*eta**2*nu+(eta/m+L*eta**2)*np.mean(eps2)
    slack=upper-np.mean(vals);assert slack>=-1e-9;slacks.append(slack)
    directional=loss(w)-eta*(h@B@h)/2+L*eta**2*nu+eta*np.mean(epsdual)+L*eta**2*np.mean(eps2)
    assert directional>=np.mean(vals)-1e-9 and directional<=upper+1e-8
    B=M*np.eye(d);h=L*w;nu=float(np.linalg.norm(B@z)**2)
    actual=np.mean([L*np.linalg.norm(w-eta*(B@(h+sign*z)))**2/2 for sign in [-1,1]])
    predicted=L*np.linalg.norm(w)**2/2-eta*(1-L*M*eta/2)*(h@B@h)+L*eta**2*nu/2
    equality.append(abs(actual-predicted));assert equality[-1]<1e-9
out['descent']={'noncommuting_random_cases':300,'minimum_discrepancy_bound_slack':min(slacks),'quadratic_equality_max_error':max(equality)}

normalization=[];transport=[];saturation=[]
for _ in range(80):
    # m=3,d=2,R=1: three active coordinates and three RMS-normalized residuals.
    n=6;p=rng.dirichlet(np.ones(4));noise=rng.normal(size=(4,n));noise-=p@noise
    h,b=rng.normal(size=(2,n))*.2;grad=noise+h+b;G=max(np.linalg.norm(grad,axis=1).max(),np.linalg.norm(h),1.)
    alpha=rng.uniform(.55,.95);a=rng.uniform(.5,1.5);ap=rng.uniform(.7,1.3)
    c=.5*rng.dirichlet(np.ones(3))*(1-alpha)*G*G;eps=.1
    def T(x):
        active=a*x[...,:3]/(np.sqrt(c+alpha*x[...,:3]**2)+eps)
        residual=ap*x[...,3:]/(np.linalg.norm(x[...,3:],axis=-1,keepdims=True)/math.sqrt(n)+eps)
        return np.concatenate([active,residual],axis=-1)
    M=max(a,ap)/eps;kappa=min(a*eps/(3*G+eps)**2,ap*eps/(3*G/math.sqrt(n)+eps)**2)
    A=np.linalg.norm(p@T(noise));cap=math.sqrt(3*a*a/alpha+n*ap*ap)
    J=min(M*G,cap);assert np.linalg.norm(T(grad),axis=1).max()<=J+1e-10
    saturation.append(J/(M*G))
    cost=np.linalg.norm(noise[:,None,:]+noise[None,:,:],axis=2)
    cons=[]
    for i in range(4):
        row=np.zeros((4,4));row[i,:]=1;cons.append(row.ravel())
    for j in range(4):
        row=np.zeros((4,4));row[:,j]=1;cons.append(row.ravel())
    lp=linprog(cost.ravel(),A_eq=np.array(cons),b_eq=np.r_[p,p],bounds=(0,None),method='highs');assert lp.success
    delta=lp.fun;assert A<=min(M*delta/2,cap)+1e-9
    lhs=h@(p@T(grad));rhs=kappa*(h@h)-(M*np.linalg.norm(b)+A)*np.linalg.norm(h)
    assert lhs>=rhs-1e-9;normalization.append(lhs-rhs);transport.append(A/(M*delta/2) if delta>0 else 0)
out['odd_field_calibration']={'cases':80,'matrix_shape':[3,2],'rank':1,'RMS_residual_included':True,'minimum_coercivity_slack':min(normalization),'max_normalized_bias_to_W1_envelope_ratio':max(transport),'maximum_saturation_to_legacy_norm_ratio':max(saturation)}

def xi(n):
    k=np.arange(n+1,dtype=float)
    logv=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+xlogy(k,k/n)+xlogy(n-k,1-k/n)
    return float(np.exp(logv).sum())

xis={n:xi(n) for n in [1,2,7,8,16,100,1000,10000]}
for n,x in xis.items():
    assert x<=n+1+1e-9
    if n>=8:assert x<=2*math.sqrt(n)+1e-9
def kl(a,b):
    if a==0:return -math.log1p(-b)
    return a*math.log(a/b)+(1-a)*math.log((1-a)/(1-b))
inverse=[]
for _ in range(200):
    a=rng.uniform(.001,.95);c=10**rng.uniform(-6,-.5);lo=a;hi=np.nextafter(1.,0.)
    for __ in range(60):
        mid=(lo+hi)/2
        if kl(a,mid)>c:hi=mid
        else:lo=mid
    b=(lo+hi)/2
    objective=lambda logg:-math.expm1(-math.exp(logg)*a-c)/(-math.expm1(-math.exp(logg)))
    opt=minimize_scalar(objective,bounds=(-20,12),method='bounded',options={'xatol':1e-12})
    inverse.append(abs(opt.fun-b));assert inverse[-1]<1e-8
for c in [1e-8,1e-4,.1]:
    b=-math.expm1(-c);assert abs(kl(0,b)-c)<1e-12
out['pac_bayes']={'exact_moment_constants':xis,'inverse_KL_Catoni_cases':200,'max_equivalence_error':max(inverse),'zero_empirical_error_formula':'1-exp(-c)','zero_empirical_error_boundary_cases':3}

gaps=[];complexity=[]
for _ in range(500):
    h=rng.uniform(0,10,size=5);d=rng.integers(1,10,size=5);n=sum(d);pi=d/n;lam=10**rng.uniform(-2,1);v0=10**rng.uniform(-2,1);mean=pi@h
    variance=pi@((h-mean)**2);gain=n/(2*lam)*(math.log1p(2*lam*v0*mean)-pi@np.log1p(2*lam*v0*h))
    lower=n*lam*v0*v0*variance/(1+2*lam*v0*max(h))**2
    upper=n*lam*v0*v0*variance/(1+2*lam*v0*min(h))**2
    assert lower<=gain+1e-9 and gain<=upper+1e-9;gaps.append(min(gain-lower,upper-gain))
    sizes=[1,2,3];dim=sum(sizes);p=blocks(orth(dim),sizes);q=blocks(orth(dim),sizes);A=orth(dim);eig=rng.uniform(0,4,dim);A=A@np.diag(eig)@A.T
    cf=lambda ps:sum(sz/(2*lam)*math.log1p(2*lam*v0*np.trace(P@A)/sz) for P,sz in zip(ps,sizes))
    dp=math.sqrt(sum(np.linalg.norm(sum(p[:j])-sum(q[:j]),'fro')**2 for j in [1,2]));bound=v0*(max(eig)-min(eig))*math.sqrt(dim)*dp
    assert abs(cf(p)-cf(q))<=bound+1e-9;assert abs(sum(np.trace(P) for P in p)-dim)<1e-10
    complexity.append(bound-abs(cf(p)-cf(q)))
    A=2*np.eye(dim);assert abs(cf(p)-cf(q))<1e-10
local_ratios=[]
for delta in [.01,.001,.0001]:
    lam=.7;v0=.4;h=np.array([1-delta,1+delta]);gain=(2*math.log1p(2*lam*v0)-sum(np.log1p(2*lam*v0*h)))/(2*lam)
    leading=2*lam*v0*v0*delta*delta/(1+2*lam*v0)**2
    local_ratios.append(gain/leading)
assert abs(local_ratios[-1]-1)<1e-5
out['directional_complexity']={'cases':500,'minimum_quantitative_Jensen_slack':min(gaps),'minimum_centered_tracking_slack':min(complexity),'isotropic_spread_bound':0,'isotropic_invariance_cases':500,'local_Jensen_gap_to_leading_term_ratios':local_ratios}

dest=Path(__file__).with_name('tight_bounds_validation.json');dest.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
