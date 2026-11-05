"""Reproducible small-instance checks of the revised theory, not benchmark wins."""
import json, math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.linalg import expm

def main():
    torch.set_num_threads(1); torch.manual_seed(42); rng=np.random.default_rng(42)
    root=Path(__file__).resolve().parents[2]/'analysis/revision_20261101'; root.mkdir(parents=True,exist_ok=True)
    b=np.array([[3,4.5,2],[0,2,1.5],[0,0,0.]])
    def obj(q): return float((b*q.T**2).sum())
    margins=[b[i,i]+b[j,j]-b[i,j]-b[j,i] for i,j in [(0,1),(0,2),(1,2)]]
    _,cols=linear_sum_assignment(-b); optimum=float(b[np.arange(3),cols].sum())
    eps=1e-4; curvature=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        s=np.zeros((3,3));s[i,j]=eps;s[j,i]=-eps
        curvature.append((obj(expm(s))+obj(expm(-s))-2*obj(np.eye(3)))/eps**2)
    # Batched full-orthogonal gradient ascent: all skew directions, 100 starts.
    q,_=torch.linalg.qr(torch.randn(100,3,3,dtype=torch.float64));bt=torch.tensor(b,dtype=q.dtype)
    for _ in range(2000):
        h=2*q*bt.T; tangent=h-q@((q.transpose(-1,-2)@h+h.transpose(-1,-2)@q)/2)
        q,_=torch.linalg.qr(q+.05*tangent)
    values=(bt*q.transpose(-1,-2).square()).sum((-1,-2)).numpy()
    # Direct finite-epsilon coordinatewise normalization example.
    g=np.array([[1.,0.]]); rot=np.array([[1,-1],[1,1.]])/math.sqrt(2); tiny=1e-6
    norm=lambda u: ((g@u)/(np.abs(g@u)+tiny))@u.T
    gauge_gap=float(np.linalg.norm(norm(np.eye(2))-norm(rot)))
    # Deterministic smooth quadratic satisfies the descent theorem exactly.
    lam=np.array([1.,2.,4.]); B=np.array([.5,1.,1.5]); L=lam.max(); bm=B.min(); bp=B.max(); eta=bm/(4*L*bp**2)
    w=np.ones(3); inequalities=[]; norms=[]
    for _ in range(1000):
        h=lam*w; old=(lam*w*w).sum()/2; nxt=w-eta*B*h; new=(lam*nxt*nxt).sum()/2
        inequalities.append(new-old+bm*eta*np.dot(h,h)/2); norms.append(float(np.dot(h,h))); w=nxt
    out=dict(seed=42,counterexample=dict(covariance1=[3,2.5,.5],covariance2=[0,2,1.5],strict_local_value=5.,global_value=optimum,swap_margins=margins,finite_difference_hessian=curvature,starts=100,global_hits=int(np.isclose(values,optimum,atol=1e-6).sum()),nonglobal_local_hits=int(np.isclose(values,5.,atol=1e-6).sum()),min_final=float(values.min()),max_final=float(values.max())),gauge=dict(epsilon=tiny,identity_update=norm(np.eye(2)).tolist(),rotated_update=norm(rot).tolist(),difference=gauge_gap),convergence=dict(steps=1000,eta=eta,max_descent_inequality_residual=float(max(inequalities)),initial_grad_squared=norms[0],final_grad_squared=norms[-1]))
    assert min(margins)>0 and max(curvature)<0 and optimum>5
    assert gauge_gap>.4 and max(inequalities)<1e-12
    (root/'theory_validation.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
