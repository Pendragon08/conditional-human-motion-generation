"""Recompute bounded projection and scalar sign alignment for batch repairs."""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets
from feature263 import CHAINS,FACE,canonicalize_z_up,encode
from g1_fk import URDFFK
from g1_joint_limit_gate import read_joint_limits
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_gpu_remaining_20261101';SIGN=ROOT/'omni_remaining_signfix_20261101';LIMIT=ROOT/'omni_remaining_limits_20261101'
HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sr=json.loads((SOURCE/'report.json').read_text());sa=json.loads((SOURCE/'audit_v2.json').read_text())
    if sha(SOURCE/'report.json')!=sa['report_sha256'] or sha(SOURCE/'records.jsonl')!=sr['records_sha256']:raise ValueError('source changed')
    inputs={r['sample_id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    pending=[]
    for folder in (SIGN,LIMIT):
        report=json.loads((folder/'report.json').read_text())
        if sha(folder/'records.jsonl')!=report['rows_sha256'] or report['source_audit_sha256']!=sha(SOURCE/'audit_v2.json'):
            raise ValueError('repair input changed')
        pending.extend((folder,x) for x in map(json.loads,(folder/'records.jsonl').read_text().splitlines()))
    if len(pending)!=26 or len({r['sample_id'] for _,r in pending})!=26:raise ValueError('unexpected repair coverage')
    limits=read_joint_limits(URDF);fk=URDFFK(URDF);skel=Skeleton(torch.as_tensor(t2m_raw_offsets,dtype=torch.float32),CHAINS,'cpu');results=[]
    with (HERE/'audit_rows.jsonl').open('x',buffering=1) as stream:
        for folder,row in pending:
            sid=row['sample_id'];src=inputs[sid]
            try:
                if row['status']!='technical_preview':raise ValueError('repair did not pass')
                original=SOURCE/src['state20'];assert sha(original)==row['parent_state_sha256']
                feature=folder/row['feature'];assert sha(feature)==row['feature_sha256']
                state=(folder/row['state']) if folder==LIMIT else original
                if folder==LIMIT:
                    assert sha(state)==row['state_sha256']
                    with np.load(original,allow_pickle=False) as a,np.load(state,allow_pickle=False) as b:
                        old=a['joint_pos'];new=b['joint_pos'];names=list(a['joint_names'])
                        for key in ('root_wxyz_xyz','sample_times','object_wxyz_xyz','source_frame_positions','joint_names'):
                            np.testing.assert_array_equal(a[key],b[key])
                        expected=np.empty_like(old)
                        for j,n in enumerate(names):
                            low,high=limits[n];expected[:,j]=[min(max(float(v),low),high) for v in old[:,j]]
                        np.testing.assert_array_equal(expected,new)
                        assert abs(float(np.max(abs(new-old)))-row['max_joint_correction_rad'])<1e-15
                with np.load(state,allow_pickle=False) as d:
                    root=d['root_wxyz_xyz'];q=d['joint_pos'];names=list(d['joint_names'])
                points=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names);raw=encode(points,foot_velocity_squared_threshold=.002)
                rotations=skel.inverse_kinematics_np(canonicalize_z_up(points),FACE,smooth_forward=True)[:,0]
                aligned=[rotations[0].copy()]
                for v in rotations[1:]:aligned.append(-v if float(np.dot(aligned[-1],v))<0 else v.copy())
                expected=[]
                for a,b in zip(aligned[1:],aligned[:-1]):
                    w,x,y,z=map(float,a);W,X,Y,Z=map(float,b)
                    expected.append(np.arcsin(np.clip(-w*Y+x*Z+y*W-z*X,-1,1)))
                with np.load(feature,allow_pickle=False) as d:features=d['features']
                np.testing.assert_array_equal(features[:,1:],raw[:,1:])
                difference=float(np.max(abs(features[:,0]-np.asarray(expected,dtype=np.float32))))
                if difference>1e-7:raise ValueError('angular verification failed')
                out={'sample_id':sid,'status':'checked','kind':'limit_projection' if folder==LIMIT else 'sign_fix',
                    'angular_max_abs_diff':difference,'training_ready':False}
            except Exception as e:
                out={'sample_id':sid,'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            results.append(out);stream.write(json.dumps(out,allow_nan=False)+'\n');stream.flush()
    out={'source_audit_sha256':sha(SOURCE/'audit_v2.json'),'sign_report_sha256':sha(SIGN/'report.json'),
        'limit_report_sha256':sha(LIMIT/'report.json'),'script_sha256':sha(Path(__file__)),'rows_sha256':sha(HERE/'audit_rows.jsonl'),
        'cases':len(results),'checked':sum(x['status']=='checked' for x in results),
        'failed':[x['sample_id'] for x in results if x['status']=='failed'],'training_ready':False,
        'limitations':['Scalar sign and projection formulas are independent; FK, IK and feature encoder are shared.',
            'Repairs are approximations, no source, semantic, physical or training acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
