"""Audit endpoint-induced feature changes and Gaussian boundary influence."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');HERE=Path(__file__).resolve().parent
INPUT=ROOT/'humanml_official_example_012314_20261031';SKEL=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen/data_loaders/humanml/common/skeleton.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=Path(sys.argv[1]);r=json.loads(report.read_text());artifact=HERE/'features.npz';assert sha(artifact)==r['artifact_sha256'] and sha(INPUT/'joints.npy')==r['joints_sha256'] and sha(SKEL)==r['skeleton_sha256']
tree=ast.parse(SKEL.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='gaussian_filter1d']
assert any(len(x.args)>=2 and isinstance(x.args[1],ast.Constant) and x.args[1].value==20 and any(k.arg=='mode' and isinstance(k.value,ast.Constant) and k.value.value=='nearest' for k in x.keywords) for x in calls)
with np.load(artifact,allow_pickle=False) as d:full=d['full'];cut=d['cut']
assert full.shape==(169,263) and cut.shape==(168,263);delta=np.abs(full[:-1]-cut)
for name,a,b in [('root',0,4),('ric',4,67),('rotation6d',67,193),('local_velocity',193,259),('foot_bits',259,263)]:
    x=delta[:,a:b];rec=r['full_vs_cut_shared_prefix'][name];assert abs(float(x.max())-rec['max_abs'])<1e-12 and abs(float(x.mean())-rec['mean_abs'])<1e-12
for threshold in (1e-7,1e-6,1e-5,1e-4):
    idx=[i for i,row in enumerate(delta) if any(float(v)>threshold for v in row)];rec=r['full_vs_cut_shared_prefix']['rows_above_'+str(threshold)]
    assert (len(idx),idx[0] if idx else None,idx[-1] if idx else None)==(rec['count'],rec['first'],rec['last'])
point=np.load(INPUT/'joints.npy',allow_pickle=False).astype(float);across=(point[:,1]-point[:,2])+(point[:,17]-point[:,16]);across/=np.linalg.norm(across,axis=1,keepdims=True)
forward=np.cross(np.array([[0.,1.,0.]]),across);offset=np.arange(-80,81);weights=np.exp(-.5*(offset/20.)**2);weights/=weights.sum()
def convolution(x):
    return np.array([sum((weights[k]*x[min(max(i+int(off),0),len(x)-1)] for k,off in enumerate(offset)),np.zeros(3)) for i in range(len(x))])
manual=convolution(forward);manual_cut=convolution(forward[:-1]);scipy=gaussian_filter1d(forward,20,axis=0,mode='nearest');scipy_cut=gaussian_filter1d(forward[:-1],20,axis=0,mode='nearest')
err=max(float(np.max(np.abs(manual-scipy))),float(np.max(np.abs(manual_cut-scipy_cut))));assert err<1e-12
influence=np.linalg.norm(manual[:-1]-manual_cut,axis=1);indices=np.flatnonzero(influence>1e-12)
out={'status':'passed','source_report_sha256':sha(report),'artifact_sha256':sha(artifact),'source_joints_sha256':sha(INPUT/'joints.npy'),'skeleton_sha256':sha(SKEL),'script_sha256':sha(Path(__file__)),'manual_vs_scipy_gaussian_max_abs':err,'kernel_sigma_frames':20,'kernel_radius_frames':80,'raw_forward_first_affected_row':int(indices[0]) if len(indices) else None,'raw_forward_max_vector_difference':float(influence.max()),'limitations':'Same saved encoder outputs audited, no independent re-execution of the full encoder; manual convolution independently checks this one input and installed SciPy boundary semantics. No raw FPS, physical correctness or training claim.'}
with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
