"""Numerical checks of derived identities and inequality constants; not proofs."""
import itertools,json
from pathlib import Path
import numpy as np
r=np.random.default_rng(42);checks={}
norm=np.linalg.norm

def qf(x):
 q,t=np.linalg.qr(x);return q*np.where(np.diag(t)<0,-1.,1.)
def polar(x):
 a,_,b=np.linalg.svd(x);return a@b

def record(name,error,tol=1e-9):
 checks[name]=dict(error=float(error),tolerance=tol,passed=bool(error<=tol))

# Projector energy, horizontal projection, and the QR remainder.
errs=[];bounds=[];align=[];transport=[]
for _ in range(100):
 u=qf(r.normal(size=(8,4)));h=r.normal(size=u.shape)
 a=u.T@h;sk=(a-a.T)/2;bd=np.zeros_like(a)
 for s in (slice(0,2),slice(2,4)):bd[s,s]=sk[s,s]
 z=h-u@((a+a.T)/2)-u@bd;z*=.03/norm(z)
 bar=qf(u+z);e=bar-u-z
 errs.append(norm(u.T@z+z.T@u))
 bounds.append(norm(e)-(1+norm(z,2)**2)*norm(z)**2)
 for s in (slice(0,2),slice(2,4)):
  old=u[:,s];new0=bar[:,s];o=polar(new0.T@old);v=new0@o;m=r.normal(size=(5,2))
  residual=norm(m@o.T@v.T-m@old.T)**2
  decomposition=norm(m@(o.T-old.T@v))**2+norm(m@old.T@(np.eye(8)-v@v.T))**2
  transport.append(abs(residual-decomposition))
  ej=norm(e[:,s]);align.append(norm(o.T-old.T@v)-(1+2/(2-ej))*ej)
record('horizontal_tangency',max(errs));record('qr_remainder_bound',max(bounds))
record('orthogonal_transport_decomposition',max(transport));record('horizontal_qr_coordinate_bound',max(align))

# Bad local maximum: coordinate Hessian agrees with (-1,-2,-1).
b=np.array([[3,4.5,2],[0,2,1.5],[0,0,0.]])
record('assignment_optimum',abs(max(b[0,i]+b[1,j] for i,j in itertools.permutations(range(3),2))-6))
c1=np.diag([3,2.5,.5]);c2=np.diag([0,2,1.5]);hh=[]
for i,j in ((0,1),(0,2),(1,2)):
 def f(t):
  q=np.eye(3);q[i,i]=q[j,j]=np.cos(t);q[j,i]=np.sin(t);q[i,j]=-np.sin(t)
  return q[:,0]@c1@q[:,0]+np.trace(q[:,:2].T@c2@q[:,:2])
 t=1e-3;hh.append((f(t)+f(-t)-2*f(0))/t**2)
record('commuting_counterexample_hessian',norm(np.array(hh)-np.array([-1,-2,-1.])),2e-6)

# Exact EMA norm and age, including severe cancellation near rho=1.
e=[]
for rho,t in itertools.product((.1,.5,.9,.999), (1,2,10,1000)):
 a=(1-rho)*rho**np.arange(t-1,-1,-1)/(1-rho**t)
 e.extend([abs(a@a-(1-rho)/(1+rho)*(1+rho**t)/(1-rho**t)),abs(a@np.arange(t-1,-1,-1)-(rho/(1-rho)-t*rho**t/(1-rho**t)))])
record('ema_exact_identities',max(e),2e-9)

# Virtual iterates for noncommuting matrix momentum and preconditioners.
e=[]
for _ in range(100):
 d=4;bt=r.normal(size=(d,d));bt*=.4/norm(bt,2)
 b0=r.normal(size=(d,d));b0=b0@b0.T+np.eye(d)
 prev=r.normal(size=d);g=r.normal(size=d);u=r.normal(size=d);w=r.normal(size=d);cp=r.normal(size=(d,d))*.01
 eta=.01;c=eta*b0@np.linalg.solve(np.eye(d)-bt,bt)
 m=bt@prev+(np.eye(d)-bt)@g+u
 z=w-cp@prev;zp=w-eta*b0@m-c@m
 rem=(cp-c)@prev-eta*b0@np.linalg.solve(np.eye(d)-bt,u)
 e.append(norm(zp-(z-eta*b0@g+rem)))
record('noncommuting_virtual_iterate_identity',max(e))

# Schur criterion and constructive left weights.
e=[]
for _ in range(100):
 d=r.uniform(0,.04,size=(4,4));b=r.uniform(0,.1,size=4);c=r.uniform(0,.1,size=4)
 inv=np.linalg.inv(np.eye(4)-d);alpha=b@inv@c+.2;eta=.1
 v=(b+.001)@inv;delta=alpha-v@c;gamma=min(delta,.001/(eta*max(v)))*.5
 mat=np.block([[np.array([[1-eta*alpha]]),eta*b[None,:]],[c[:,None],d]])
 left=np.r_[1,eta*v];e.append(np.max(left@mat-(1-gamma*eta)*left))
 assert max(abs(np.linalg.eigvals(mat)))<1
record('constructive_schur_left_weight',max(e))

# Gaussian KL: flag expression versus generic full covariance expression.
e=[]
for _ in range(100):
 u=qf(r.normal(size=(6,6)));sig=np.exp(r.normal(size=3));qs=(1,2,3);idx=np.repeat(range(3),qs)
 cov=u@np.diag(sig[idx]**2)@u.T;w=r.normal(size=(3,6));s0=.7
 flag=.5*norm(w)**2/s0**2+1.5*sum(qs[j]*(sig[j]**2/s0**2-1-np.log(sig[j]**2/s0**2)) for j in range(3))
 full=.5*(norm(w)**2/s0**2+3*np.trace(cov)/s0**2-18+18*np.log(s0**2)-3*np.linalg.slogdet(cov)[1]);e.append(abs(flag-full))
record('flag_posterior_gaussian_kl',max(e),1e-7)

# Smooth nonconvex one-step descent with a current-noise-dependent discrepancy.
e=[];L=1.2;lo=.5;hi=2.;eta=lo/(8*L*hi**2)
F=lambda x:np.sum(.1*x*x+1-np.cos(x))
for _ in range(100):
 w=r.normal(size=4);h=.2*w+np.sin(w);b=np.diag(r.uniform(lo,hi,size=4));noise=r.normal(size=4);ds=[];ee=[]
 for sign in (-1,1):
  z=sign*noise;err=.03*z+.01*w;ds.append(F(w-eta*(b@(h+z)+err)));ee.append(norm(err)**2)
 rhs=F(w)-lo*eta*norm(h)**2/2+L*hi**2*eta**2*norm(noise)**2+(eta/lo+L*eta**2)*np.mean(ee)
 e.append(np.mean(ds)-rhs)
record('smooth_nonconvex_conditional_descent',max(e))
report=dict(seed=42,checks=checks,all_passed=all(v['passed'] for v in checks.values()))
out=Path(__file__).resolve().parents[2]/'analysis/revision_20261101/rigid_math_validation.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert report['all_passed']
