"""Independently compute polyphase FIR by explicit summation, including line extension."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.signal import firwin

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
SRC=ROOT/'humanml_mirror_time_witness_v2_20261030/002511_joint_witness.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def direct(x,fps,t):
    x=np.asarray(x,dtype=np.float64);s=x.shape;v=x.reshape(len(x),-1); n=len(v)
    h=2*firwin(61,1/3,window=('kaiser',5.0)); count=math.ceil(n*2/3);out=np.empty((count,v.shape[1]))
    slope=(v[-1]-v[0])/(n-1)
    for k in range(count):
        sample=3*k;lo=math.ceil((sample-30)/2);hi=math.floor((sample+30)/2)
        ids=np.arange(lo,hi+1);vals=v[np.clip(ids,0,n-1)].copy()
        vals[ids<0]=v[0]+ids[ids<0,None]*slope
        vals[ids>=n]=v[-1]+(ids[ids>=n,None]-(n-1))*slope
        out[k]=(h[30+sample-2*ids,None]*vals).sum(axis=0)
    tt=np.arange(count)/(fps*2/3)
    return np.column_stack([np.interp(t,tt,out[:,i]) for i in range(v.shape[1])]).reshape((len(t),)+s[1:])
def amp(x,t):
    m=(t>=.5)&(t<=t[-1]-.5);a=np.stack([np.sin(12*np.pi*t[m]),np.cos(12*np.pi*t[m]),np.ones(m.sum())],axis=-1)
    b=np.linalg.lstsq(a,x[m],rcond=None)[0];return float(np.hypot(b[0],b[1]))
def main():
    r=json.loads((HERE/'report.json').read_text());assert sha(SRC)==r['source_sha256'] and sha(HERE/'run.py')==r['script_sha256']
    d=np.load(SRC,allow_pickle=False);t=d['output_times'];x=d['original_joints_humanml'];mir=d['mirrored_joints_humanml'];fps=r['input_fps_assumed_from_reference']
    y=direct(x,fps,t);z=direct(mir,fps,t);v=np.linalg.norm(y-d['original_joint_linear_20hz'],axis=2);edges=(t<.5)|(t>t[-1]-.5)
    for key,val in [('mean',v.mean()),('median',np.median(v)),('max',v.max()),('edge_mean',v[edges].mean()),('interior_mean',v[~edges].mean())]:assert abs(r['joint_distance_fir_vs_linear_m'][key]-val)<1e-12,(key,val)
    reflected=y.copy();reflected[...,0]*=-1
    left=(1,4,7,10,13,16,18,20);right=(2,5,8,11,14,17,19,21)
    reflected[:,left,:]=y[:,right,:]*[-1,1,1];reflected[:,right,:]=y[:,left,:]*[-1,1,1]
    mirror_diff=float(np.max(np.abs(reflected-z)));assert mirror_diff<1e-10
    ts=np.arange(301)/30;tx=np.arange(201)/20;sin=np.sin(28*np.pi*ts)
    af=amp(direct(sin,30,tx),tx);al=amp(np.interp(tx,ts,sin),tx)
    assert abs(af-r['synthetic_14hz_alias_6hz']['fir_amplitude'])<1e-12 and abs(al-r['synthetic_14hz_alias_6hz']['linear_amplitude'])<1e-12
    result={'source_sha256':sha(SRC),'primary_report_sha256':sha(HERE/'report.json'),'independent_script_sha256':sha(Path(__file__)),
            'mirror_max_m':mirror_diff,'synthetic_linear_amplitude':al,'synthetic_fir_amplitude':af,
            'check':'Explicit 61-tap sum, linear edge extension and independent mirror. Same filter specification and same input, not independent physical truth.'}
    with (HERE/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
