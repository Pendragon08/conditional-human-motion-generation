"""Non-training 30->20Hz joint-position antialias comparison on existing witness."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from humanml_joint_mirroring import mirror_humanml

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'humanml_mirror_time_witness_v2_20261030/002511_joint_witness.npz'
OUT=Path(__file__).resolve().parent
FPS=29.9999942779541
EXPECTED='9dc188d1d387bd8e208cd928ca6aebaa5e6fb0e84c8a58e05a313c4b961b6d16'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def filtered(x,fps,times):
    # Rational approximation 2/3, then interpolation back to the strict 20 Hz grid.
    a=np.asarray(x,dtype=np.float64);flat=a.reshape(len(a),-1)
    y=resample_poly(flat,2,3,axis=0,window=('kaiser',5),padtype='line')
    yt=np.arange(len(y))/(fps*2/3)
    if times[-1]>(len(a)-1)/fps+1e-12 or times[-1]>yt[-1]:raise ValueError('grid exceeds support')
    return np.column_stack([np.interp(times,yt,y[:,i]) for i in range(flat.shape[1])]).reshape((len(times),)+a.shape[1:])
def amplitude(x,t,hz):
    use=(t>=0.5)&(t<=t[-1]-0.5)
    basis=np.column_stack((np.sin(2*np.pi*hz*t[use]),np.cos(2*np.pi*hz*t[use]),np.ones(use.sum())))
    coef=np.linalg.lstsq(basis,x[use],rcond=None)[0]
    return float(np.hypot(coef[0],coef[1]))
def main():
    if sha(SOURCE)!=EXPECTED:raise ValueError('witness hash changed')
    d=np.load(SOURCE,allow_pickle=False);t=d['output_times'];x=d['original_joints_humanml'];m=d['mirrored_joints_humanml'];linear=d['original_joint_linear_20hz']
    assert len(x)==200 and len(t)==133 and np.max(np.abs(np.diff(t)-.05))<1e-12
    fir=filtered(x,FPS,t);mf=filtered(m,FPS,t)
    comm=float(np.max(np.abs(mf-mirror_humanml(fir))))
    diff=np.linalg.norm(fir-linear,axis=2);edge=(t<.5)|(t>t[-1]-.5)
    ts=np.arange(301)/30.0;tx=np.arange(201)/20.0;synthetic=np.sin(2*np.pi*14*ts)
    synthetic_fir=filtered(synthetic,30,tx);synthetic_linear=np.interp(tx,ts,synthetic)
    linamp=amplitude(synthetic_linear,tx,6);firamp=amplitude(synthetic_fir,tx,6)
    if not (comm<1e-10 and firamp<linamp*.1 and t[-1]<=(len(x)-1)/FPS):raise ValueError('synthetic, mirror or time check failed')
    output={'source_sha256':EXPECTED,'script_sha256':sha(Path(__file__)),'mirror_module_sha256':sha(Path(__file__).with_name('humanml_joint_mirroring.py')),
      'input_fps_assumed_from_reference':FPS,'input_count':len(x),'output_count':len(t),'last_grid_s':float(t[-1]),
      'source_last_s':float((len(x)-1)/FPS),'rational_filter_rate_hz':FPS*2/3,'filter':'scipy resample_poly 2/3, Kaiser beta 5, line padding; linear interpolation to exact 20Hz grid',
      'joint_distance_fir_vs_linear_m':{'mean':float(diff.mean()),'median':float(np.median(diff)),'max':float(diff.max()),'edge_mean':float(diff[edge].mean()),'interior_mean':float(diff[~edge].mean())},
      'mirror_commutation_max_m':comm,'synthetic_14hz_alias_6hz':{'linear_amplitude':linamp,'fir_amplitude':firamp,'ratio':firamp/linamp},
      'limitations':'One short SSM clip; source FPS/model identity relies on prior witness. Joint-space FIR and boundary padding are assumptions; 14 Hz synthetic test is not source motion truth. Does not test pose rotations, bone constraints, contacts, robot physics or actual reference truth; no training data.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(output,sort_keys=True))
if __name__=='__main__':main()
