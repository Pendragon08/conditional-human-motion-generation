"""Independent NumPy decoding and frame-difference check for one official example."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=Path(sys.argv[1]);r=json.loads(report.read_text());jp=HERE/'joints.npy';fp=HERE/'features.npy';nb=HERE/'motion_representation.ipynb'
assert sha(jp)==r['joints_sha256'] and sha(fp)==r['features_sha256'] and sha(nb)==r['notebook_sha256']
f=np.load(fp,allow_pickle=False).astype(np.float64);target=np.load(jp,allow_pickle=False).astype(np.float64);n=len(f)
angle=np.zeros(n);angle[1:]=np.cumsum(f[:-1,0]);c=np.cos(2*angle);s=np.sin(2*angle)
def inverse_y(v,index):
    x,y,z=v[...,0],v[...,1],v[...,2];return np.stack([c[index]*x-s[index]*z,y,s[index]*x+c[index]*z],axis=-1)
# Test known inverse 90 degree Y rotation without using upstream quaternion code.
assert np.allclose(np.array([math.cos(math.pi/2),0,math.sin(math.pi/2)]),[0,0,1],atol=1e-15)
root=np.zeros((n,3));root[:,1]=f[:,3]
for i in range(1,n):
    d=inverse_y(np.array([f[i-1,1],0,f[i-1,2]]),i);root[i,[0,2]]=root[i-1,[0,2]]+d[[0,2]]
out=np.zeros((n,22,3));out[:,0]=root
for i in range(n):
    local=f[i,4:67].reshape(21,3);out[i,1:]=inverse_y(local,i);out[i,1:,0]+=root[i,0];out[i,1:,2]+=root[i,2]
recover_error=float(np.max(np.abs(out-target)));assert recover_error<1e-5
velocity=[]
for i in range(n-1):velocity.append(inverse_y(f[i,193:259].reshape(22,3),i))
velocity_error=float(np.max(np.abs(np.array(velocity)-np.diff(target,axis=0))));assert velocity_error<1e-5
foot=[]
for i in range(n-1):
    foot.append([sum((target[i+1,k,a]-target[i,k,a])**2 for a in range(3))<.002 for k in (7,10,8,11)])
assert int(np.count_nonzero(np.array(foot)!=f[:-1,-4:]))==r['foot_differences']
out={'status':'passed','source_report_sha256':sha(report),'joints_sha256':sha(jp),'features_sha256':sha(fp),'script_sha256':sha(Path(__file__)),'numpy_recovery_max_abs_m':recover_error,'frame_differences_checked':n-1,'world_velocity_max_abs_m_per_frame':velocity_error,'last_velocity_row_unchecked':True,'foot_decisions_checked':(n-1)*4,'limitations':'Independent decoder and frame differences on a derived published example; no independent original raw joint trajectory, real FPS or physical foot contacts.'}
with Path(sys.argv[2]).open('x',encoding='utf8') as fh:json.dump(out,fh,indent=2,allow_nan=False)
