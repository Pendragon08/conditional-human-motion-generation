"""Experimental 60 Hz robot state FIR and strict 20 Hz timestamp conversion.

Quaternion component filtering is approximate; it does not establish source
timing, contacts, dynamics or the unpublished author preprocessing method.
"""
import math
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation, Slerp


def resample_robot_60_to_20(qpos, source_times, *, target_fps=20.0):
    state = np.asarray(qpos,dtype=np.float64)
    times = np.asarray(source_times,dtype=np.float64)
    if (state.ndim != 2 or state.shape[1] != 34 or len(state) < 7 or times.shape != (len(state),) or
        not np.isfinite(state).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
        not math.isfinite(target_fps) or target_fps != 20.):
        raise ValueError('expected finite robot state and strictly increasing source times at target 20 Hz')
    dt = float(np.median(np.diff(times))); fps = 1. / dt
    if abs(fps-60.) > .001 or not np.allclose(np.diff(times),dt,rtol=0,atol=1e-10):
        raise ValueError('expected uniform reference rate approximately 60 Hz')
    if not np.allclose(np.linalg.norm(state[:,:4],axis=1),1.,atol=1e-4,rtol=0):
        raise ValueError('nonunit root quaternion')
    x=state.copy(); flips=np.where(np.sum(x[1:,:4]*x[:-1,:4],axis=1)<0,-1.,1.)
    x[:,:4]*=np.r_[1.,np.cumprod(flips)][:,None]
    fraction=np.arange(len(x),dtype=float)[:,None]/(len(x)-1); delta=x[-1]-x[0]
    residual=x-(x[0]+fraction*delta)
    poly=resample_poly(residual,1,3,axis=0,window=('kaiser',5.),padtype='line')
    poly += x[0] + (np.arange(len(poly),dtype=float)*3/(len(x)-1))[:,None]*delta
    # The kth polyphase output is centered on source sample 3*k. Read the
    # actual timestamp rather than accumulating a rounded median interval.
    poly_times=times[np.arange(len(poly),dtype=np.int64)*3].copy()
    stop=min(float(times[-1]),float(poly_times[-1]))
    count=int(np.floor((stop-float(times[0]))*target_fps+1e-10))+1
    grid=times[0]+np.arange(count,dtype=np.float64)/target_fps
    if grid[-1] > stop:
        grid=grid[:-1]; count-=1
    if count < 2 or grid[-1] > stop:
        raise ValueError('invalid target grid')
    norms=np.linalg.norm(poly[:,:4],axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('degenerate filtered quaternion')
    poly[:,:4] /= norms[:,None]
    poly_flips=np.where(np.sum(poly[1:,:4]*poly[:-1,:4],axis=1)<0,-1.,1.)
    poly[:,:4]*=np.r_[1.,np.cumprod(poly_flips)][:,None]
    result=np.empty((count,34),dtype=np.float64)
    result[:,4:]=np.column_stack([np.interp(grid,poly_times,poly[:,j]) for j in range(4,34)])
    rotations=Rotation.from_quat(poly[:,[1,2,3,0]])
    result[:,:4]=Slerp(poly_times,rotations)(grid).as_quat()[:,[3,0,1,2]]
    result[0]=x[0]
    if not np.isfinite(result).all():
        raise ValueError('invalid output')
    return {'states':result,'times':grid,'source_fps':fps,'polyphase_times':poly_times,
            'discarded_tail_s':float(times[-1]-grid[-1])}
