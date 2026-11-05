"""Experimental 30→20 Hz FIR resampling of translation and axis-angle joints.

Quaternion component filtering is an SO(3) approximation. Sampling-time
identity and source reference rate must be established independently.
"""
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation, Slerp


def fir_resample(translation, pose, source_fps, target_fps=20.0):
    p=np.asarray(translation,dtype=np.float64);rv=np.asarray(pose,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or rv.ndim!=2 or rv.shape[1]%3 or len(p)!=len(rv) or len(p)<4:
        raise ValueError('invalid motion dimensions')
    if not np.isfinite(p).all() or not np.isfinite(rv).all() or not np.isfinite(source_fps) or not np.isfinite(target_fps):
        raise ValueError('nonfinite input')
    if not 0<target_fps<source_fps:raise ValueError('only downsampling supported')
    ratio=Fraction(float(target_fps/source_fps)).limit_denominator(1000)
    if abs(float(ratio)-target_fps/source_fps)>1e-5:raise ValueError('unsupported rate ratio')
    n,joints=len(p),rv.shape[1]//3
    quat=Rotation.from_rotvec(rv.reshape(-1,3)).as_quat().reshape(n,joints,4)
    signs=np.where((quat[1:]*quat[:-1]).sum(axis=2)<0,-1.0,1.0)
    quat=quat*np.concatenate([np.ones((1,joints)),np.cumprod(signs,axis=0)],axis=0)[:,:,None]
    channels=np.concatenate([p,quat.reshape(n,-1)],axis=1)
    trend=(channels[-1]-channels[0])
    residual=channels-(channels[0]+np.arange(n)[:,None]/(n-1)*trend)
    sampled=resample_poly(residual,ratio.numerator,ratio.denominator,axis=0,window=('kaiser',5.0),padtype='line')
    actual_rate=source_fps*ratio.numerator/ratio.denominator
    times=np.arange(len(sampled))/actual_rate
    sampled+=channels[0]+(times*source_fps/(n-1))[:,None]*trend
    duration=(n-1)/source_fps
    count=int(np.floor(duration*target_fps+1e-10))+1
    desired=np.arange(count)/target_fps
    if count<2 or desired[-1]>times[-1]+1e-10:raise ValueError('output grid unavailable')
    sampled[0]=channels[0]
    sampled_q=sampled[:,3:].reshape(len(sampled),joints,4)
    norms=np.linalg.norm(sampled_q,axis=2,keepdims=True)
    if np.any(norms<1e-8):raise ValueError('filtered quaternion near zero')
    sampled_q/=norms
    positions=np.column_stack([np.interp(desired,times,sampled[:,i]) for i in range(3)])
    out=np.empty((count,joints,3))
    for j in range(joints):
        out[:,j]=Slerp(times,Rotation.from_quat(sampled_q[:,j,:]))(desired).as_rotvec()
    return positions,out.reshape(count,-1),{'source_fps':source_fps,'filter_rate':actual_rate,
        'target_fps':target_fps,'up':ratio.numerator,'down':ratio.denominator,
        'source_duration_s':duration,'output_last_time_s':float(desired[-1])}
