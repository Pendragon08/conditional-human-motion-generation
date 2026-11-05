"""Exploratory overlap blend; does not prove physical or contact continuity."""
import numpy as np

def blend_windows(windows):
    if not windows:raise ValueError('empty windows')
    weights=np.array([1.,2.,3.,4.,3.,2.,1.])
    accum={}; names=None
    for w in windows:
        frames=np.asarray(w['frames']); root=np.asarray(w['root_wxyz_xyz'],dtype=float);q=np.asarray(w['joint_pos'],dtype=float)
        current=tuple(w['joint_names'])
        if names is None:names=current
        if current!=names or len(names)!=27 or len(set(names))!=27:raise ValueError('joint name mismatch')
        if frames.shape!=(7,) or frames.dtype.kind not in 'iu' or root.shape!=(7,7) or q.shape!=(7,27) or np.any(np.diff(frames)!=1):raise ValueError('bad window')
        if not np.isfinite(root).all() or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(root[:,:4],axis=1),1.,atol=1e-4,rtol=0):raise ValueError('invalid values')
        for i,frame in enumerate(frames):
            f=int(frame);weight=weights[i]; value=np.r_[root[i],q[i]]
            if f not in accum:
                accum[f]={'sum':np.zeros(34), 'weight':0., 'reference':root[i,:4].copy(),'first':value.copy()}
            a=accum[f]; value=value.copy()
            if np.dot(value[:4],a['reference'])<0:value[:4]*=-1
            a['sum']+=weight*value;a['weight']+=weight
    frames=np.array(sorted(accum),dtype=np.int64)
    if np.any(np.diff(frames)!=1):raise ValueError('frame coverage has a gap')
    first=np.stack([accum[int(f)]['first'] for f in frames]); blended=np.stack([accum[int(f)]['sum']/accum[int(f)]['weight'] for f in frames])
    norms=np.linalg.norm(blended[:,:4],axis=1)
    if np.any(norms<1e-8):raise ValueError('degenerate quaternion blend')
    blended[:,:4]/=norms[:,None]
    for trajectory in (first,blended):
        flips=np.where(np.sum(trajectory[1:,:4]*trajectory[:-1,:4],axis=1)<0,-1.,1.)
        trajectory[:,:4]*=np.r_[1.,np.cumprod(flips)][:,None]
    return {'frames':frames,'first':first,'blended':blended,'joint_names':names}
