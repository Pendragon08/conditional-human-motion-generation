"""Bounded horizontal reference translation; diagnostic, not a learned controller.

The input velocities are simulation/geometry proxies. Contact anchoring itself
is an established idea, not a novelty claim. The endpoint-preservation ablation
tests whether this limited reference-side intervention is informative here.
"""
import numpy as np


def translation(velocity, support, start, dt, preserve_endpoint, max_shift=.10):
    velocity=np.asarray(velocity,dtype=np.float64)
    support=np.asarray(support,dtype=bool)
    t,feet,_=velocity.shape
    assert support.shape==(t,feet) and 0<start<t and dt>0
    assert np.isfinite(velocity).all()
    # Velocity matching + displacement regularization + temporal smoothness.
    d1=np.diff(np.eye(t),axis=0)/dt
    d2=np.diff(np.eye(t),n=2,axis=0)/(dt*dt)
    weight=(support[1:] & support[:-1]).astype(np.float64)
    mean=(velocity[1:]*weight[:,:,None]).sum(axis=1)
    h=d1.T @ (weight.sum(axis=1)[:,None]*d1)+5*np.eye(t)+.005*(d2.T@d2)
    b=d1.T @ mean
    active=np.arange(start+1,t-1 if preserve_endpoint else t)
    out=np.zeros((t,2),dtype=np.float64)
    if len(active):
        a=h[np.ix_(active,active)];g=b[active]
        step=1./np.linalg.eigvalsh(a)[-1]
        x=np.zeros((len(active),2),dtype=np.float64)
        for _ in range(400):
            x=x-step*(a@x+g)
            n=np.linalg.norm(x,axis=1,keepdims=True)
            x=x*np.minimum(1.,max_shift/np.maximum(n,1e-12))
        out[active]=x
    assert np.max(np.linalg.norm(out,axis=1))<=max_shift+1e-12
    assert not out[:start+1].any()
    if preserve_endpoint:assert not out[-1].any()
    return out


def numeric_checks():
    v=np.zeros((120,4,2));s=np.ones((120,4),bool)
    assert not translation(v,s,60,1/30,True).any()
    v[:,:,0]=.2
    p=translation(v,s,60,1/30,True);u=translation(v,s,60,1/30,False)
    assert p[60,0]==p[-1,0]==0 and u[-1,0]<0
    # Constant velocity integrates to a fixed displacement: pinning both
    # ends cannot cancel it everywhere. A zero constrained optimum is valid.
    assert np.max(np.abs(p))<1e-12
    v[90:,:,0]=-.2
    varied=translation(v,s,60,1/30,True)
    assert np.max(np.linalg.norm(varied,axis=1))>1e-3
    assert not varied[60].any() and not varied[-1].any()
    assert not translation(v,np.zeros_like(s),60,1/30,True).any()
    return dict(status='PASS_NUMERIC_CORE',scope='Synthetic stationary, translating and unsupported fixtures only; not efficacy')


if __name__=='__main__':print(numeric_checks())
