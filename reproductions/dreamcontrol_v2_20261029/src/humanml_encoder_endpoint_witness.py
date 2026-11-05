"""Measure one input endpoint's effect on the experimental 263D encoder."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from feature263 import canonicalize_z_up,encode
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=ROOT/'humanml_official_example_012314_20261031';TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def groups(a,b):
    d=np.abs(a-b);out={}
    for name,left,right in [('root',0,4),('ric',4,67),('rotation6d',67,193),('local_velocity',193,259),('foot_bits',259,263)]:
        x=d[:,left:right];out[name]={'max_abs':float(x.max()),'mean_abs':float(x.mean()),'first_20_max_abs':float(x[:20].max()),'last_20_max_abs':float(x[-20:].max()),'max_row':int(np.unravel_index(x.argmax(),x.shape)[0])}
    for threshold in (1e-7,1e-6,1e-5,1e-4):
        idx=np.flatnonzero(np.any(d>threshold,axis=1));out['rows_above_'+str(threshold)]={'count':len(idx),'first':int(idx[0]) if len(idx) else None,'last':int(idx[-1]) if len(idx) else None}
    return out
def main():
    j=INPUT/'joints.npy';f=INPUT/'features.npy';r=INPUT/'report.json';assert sha(j)=='739df5e2e025d21ae5ce01ea7e0a8d3038448a46e88865d95a638e6242ee8b6b' and sha(f)=='6e6fc0204ea9e48abdf94f84e8be22b19a1f1e955c7b98e025b1d702f6fe96a0'
    joints=np.load(j,allow_pickle=False);published=np.load(f,allow_pickle=False);assert joints.shape==(170,22,3) and published.shape==(170,263)
    # Invert the encoder's [x,z,-y] transform; this is a coordinate check only.
    z=joints[:,:,[0,2,1]].copy();z[:,:,1]*=-1
    full=encode(z,foot_velocity_squared_threshold=.002);cut=encode(z[:-1],foot_velocity_squared_threshold=.002)
    assert full.shape==(169,263) and cut.shape==(168,263)
    canonical_error=float(np.max(np.abs(canonicalize_z_up(z)[:-1]-canonicalize_z_up(z[:-1]))))
    artifact=Path(sys.argv[1]);
    with artifact.open('xb') as stream:np.savez_compressed(stream,full=full,cut=cut)
    out={'input_contract_report_sha256':sha(r),'joints_sha256':sha(j),'published_features_sha256':sha(f),'extractor_sha256':sha(TRAIN/'data_loaders/humanml/scripts/motion_process.py'),'skeleton_sha256':sha(TRAIN/'data_loaders/humanml/common/skeleton.py'),'encoder_sha256':sha(Path(sys.modules['feature263'].__file__)),'script_sha256':sha(Path(__file__)),'artifact_sha256':sha(artifact),'input_frame_counts':[170,169],'output_feature_counts':[169,168],'canonical_shared_prefix_max_abs_m':canonical_error,'full_vs_cut_shared_prefix':groups(full[:-1],cut),'reencoded_full_vs_published_prefix':groups(full,published[:-1]),'limitations':'Single author-published derived human example, not raw source, robot data or validation; the experimental encoder skips official skeleton rescaling. One removed endpoint, no claim real-time acausal error magnitude, FPS, physics or training readiness.'}
    with Path(sys.argv[2]).open('x',encoding='utf8') as stream:json.dump(out,stream,indent=2,allow_nan=False)
if __name__=='__main__':main()
