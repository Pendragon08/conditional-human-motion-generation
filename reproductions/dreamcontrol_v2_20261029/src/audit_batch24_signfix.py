"""Independently compute root sign alignment and angular increments by scalars."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets
from feature263 import CHAINS,FACE,canonicalize_z_up,encode
from g1_fk import URDFFK
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;SOURCE=ROOT/'omni_gpu_batch24_20261101';FIXED=ROOT/'omni_batch24_projection_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=json.loads((HERE/'report.json').read_text());assert sha(HERE/'rows.jsonl')==report['rows_sha256']
    rows=[json.loads(x) for x in (HERE/'rows.jsonl').read_text().splitlines()]; source={x['sample_id']:x for x in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    assert len(rows)==24 and len({r['sample_id'] for r in rows})==24
    skel=Skeleton(torch.as_tensor(t2m_raw_offsets,dtype=torch.float32),CHAINS,'cpu');fk=URDFFK(URDF);maxdiff=0.;changed=[]
    for row in rows:
        if row['status']!='technical_checks_passed':raise ValueError('source validation failed')
        sid=row['sample_id'];path=(FIXED/(sid+'_state.npz')) if row['state_variant']=='projected' else (SOURCE/source[sid]['state20'])
        if sha(path)!=row['state_sha256']:raise ValueError('state hash changed')
        with np.load(path,allow_pickle=False) as f:root=f['root_wxyz_xyz'];joints=f['joint_pos'];names=list(f['joint_names'])
        points=fk.forward(root[:,4:],root[:,[1,2,3,0]],joints,names);raw=encode(points,foot_velocity_squared_threshold=.002)
        rotations=skel.inverse_kinematics_np(canonicalize_z_up(points),FACE,smooth_forward=True)[:,0]
        # Explicit scalar sign tracking and quaternion multiplication, independent
        # of the fix module's vectorized sign alignment and qmul helper.
        fixed=[rotations[0].copy()]
        for v in rotations[1:]:fixed.append(-v if float(np.dot(fixed[-1],v))<0 else v.copy())
        fixed=np.array(fixed);expected=[]
        for a,b in zip(fixed[1:],fixed[:-1]):
            w,x,y,z=map(float,a);W,X,Y,Z=map(float,b)
            expected.append(np.arcsin(np.clip(-w*Y+x*Z+y*W-z*X,-1,1)))
        artifact=row['signfixed_feature_artifact']
        if artifact:
            path=HERE/artifact['path'];assert sha(path)==artifact['sha256']
            with np.load(path,allow_pickle=False) as f:features=f['features']
            changed.append(sid)
        else:features=raw
        assert np.array_equal(raw[:,1:],features[:,1:])
        diff=float(np.max(abs(features[:,0]-np.asarray(expected,dtype=np.float32))));maxdiff=max(maxdiff,diff)
        if diff>1e-7:raise ValueError('scalar angular channel mismatch')
    result={'source_report_sha256':sha(HERE/'report.json'),'rows_sha256':sha(HERE/'rows.jsonl'),
        'script_sha256':sha(Path(__file__)),'cases':len(rows),'changed_cases':changed,
        'max_scalar_angular_difference':maxdiff,'training_ready':False,
        'limitations':['Skeleton IK, base feature encoder and FK are shared; no independent optimizer.',
            'Quaternion sign adjustment is an approximation and is not a text, contact or physics acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
