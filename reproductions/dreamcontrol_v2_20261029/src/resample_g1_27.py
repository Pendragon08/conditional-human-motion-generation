"""Experimental 120->20 Hz antialiasing for G1 27DoF root/joint arrays."""
import numpy as np
from scipy.signal import resample_poly

def resample_27(qpos,source_fps=120.,target_fps=20.):
    """Input (T,34) WXYZ + XYZ + 27 angles; returns exact target-time samples.

    Quaternion component FIR+renormalization approximates SO(3); use only for
    diagnostics until contact, rotation and boundary effects are validated.
    """
    q=np.asarray(qpos,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=34 or len(q)<7 or not np.isfinite(q).all():raise ValueError('expected finite (T,34) with at least 7 frames')
    if source_fps!=120. or target_fps!=20.:raise ValueError('only 120->20 is supported')
    norms=np.linalg.norm(q[:,:4],axis=1)
    if not np.allclose(norms,1.,atol=1e-4,rtol=0):raise ValueError('nonunit quaternion')
    x=q.copy(); signs=np.where(np.sum(x[1:,:4]*x[:-1,:4],axis=1)<0,-1.,1.)
    x[:,:4]*=np.r_[1.,np.cumprod(signs)][:,None]
    f=np.arange(len(x),dtype=float)[:,None]/(len(x)-1);delta=x[-1]-x[0]
    residual=x-(x[0]+f*delta)
    out=resample_poly(residual,1,6,axis=0,window=('kaiser',5.0),padtype='line')
    count=int(np.floor((len(x)-1)/6))+1;out=out[:count].copy()
    out+=x[0]+(np.arange(count,dtype=float)*6/(len(x)-1))[:,None]*delta
    out[0]=x[0]
    if (count-1)*6==len(x)-1:out[-1]=x[-1]
    n=np.linalg.norm(out[:,:4],axis=1)
    if np.any(n<1e-8):raise ValueError('degenerate quaternion')
    out[:,:4]/=n[:,None]
    if not np.isfinite(out).all():raise ValueError('nonfinite output')
    return out
