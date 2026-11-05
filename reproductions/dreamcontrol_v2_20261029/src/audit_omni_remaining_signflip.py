"""Scalar quaternion recheck for isolated variant encoding repairs."""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets
from feature263 import CHAINS,FACE,canonicalize_z_up,encode
from g1_fk import URDFFK

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_variants_gpu_20261101';SNAPSHOT=SOURCE
REPAIR=ROOT/'omni_variants_remaining_signfix_20261101';HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    snap=json.loads((SNAPSHOT/'report.json').read_text());audit=json.loads((SNAPSHOT/'audit_v2.json').read_text())
    repair=json.loads((REPAIR/'report.json').read_text())
    if sha(SNAPSHOT/'report.json')!=audit['report_sha256'] or sha(SNAPSHOT/'records.jsonl')!=snap['records_sha256'] or sha(SNAPSHOT/'audit_rows_v2.jsonl')!=audit['audit_rows_sha256'] or repair['source_audit_sha256']!=sha(SNAPSHOT/'audit_v2.json') or sha(REPAIR/'records.jsonl')!=repair['rows_sha256']:
        raise ValueError('snapshot or repair evidence changed')
    inputs={r['sample_id']:r for r in map(json.loads,(SNAPSHOT/'records.jsonl').read_text().splitlines())}
    rows=[r for r in map(json.loads,(REPAIR/'records.jsonl').read_text().splitlines()) if r['status']=='technical_preview']
    if len(rows)!=repair['technical_previews'] or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('repair coverage mismatch')
    fk=URDFFK(URDF);skel=Skeleton(torch.as_tensor(t2m_raw_offsets,dtype=torch.float32),CHAINS,'cpu');results=[]
    with (HERE/'records.jsonl').open('x',buffering=1) as out:
        for row in rows:
            sid=row['sample_id'];src=inputs[sid]
            try:
                state=SOURCE/src['state20'];feature=REPAIR/row['feature']
                if sha(state)!=row['parent_state_sha256'] or sha(feature)!=row['feature_sha256']:raise ValueError('artifact changed')
                with np.load(state,allow_pickle=False) as d:root=d['root_wxyz_xyz'];q=d['joint_pos'];names=list(d['joint_names'])
                points=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names);raw=encode(points,foot_velocity_squared_threshold=.002)
                rotations=skel.inverse_kinematics_np(canonicalize_z_up(points),FACE,smooth_forward=True)[:,0]
                aligned=[rotations[0].copy()]
                for v in rotations[1:]:aligned.append(-v if float(np.dot(aligned[-1],v))<0 else v.copy())
                angular=[]
                for a,b in zip(aligned[1:],aligned[:-1]):
                    w,x,y,z=map(float,a);W,X,Y,Z=map(float,b)
                    angular.append(np.arcsin(np.clip(-w*Y+x*Z+y*W-z*X,-1,1)))
                with np.load(feature,allow_pickle=False) as d:features=d['features']
                np.testing.assert_array_equal(features[:,1:],raw[:,1:])
                error=float(np.max(abs(features[:,0]-np.asarray(angular,dtype=np.float32))))
                if error>1e-7:raise ValueError('angular scalar check failed')
                result={'sample_id':sid,'status':'checked','angular_max_abs_diff':error,'training_ready':False}
            except Exception as e:
                result={'sample_id':sid,'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            results.append(result);out.write(json.dumps(result,allow_nan=False)+'\n')
    result={'snapshot_audit_sha256':sha(SNAPSHOT/'audit_v2.json'),'repair_report_sha256':sha(REPAIR/'report.json'),
            'script_sha256':sha(Path(__file__)),'rows_sha256':sha(HERE/'records.jsonl'),'cases':len(results),
            'checked':sum(x['status']=='checked' for x in results),'failed':[x['sample_id'] for x in results if x['status']=='failed'],
            'training_ready':False,'limitations':['Scalar quaternion arithmetic independent; same FK, IK and feature encoder.',
                'Quarantined correction remains non-author approximation with no semantic, source, physical or training acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
