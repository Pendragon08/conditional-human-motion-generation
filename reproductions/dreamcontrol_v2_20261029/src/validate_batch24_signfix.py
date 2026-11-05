"""Evaluate sign-fixed encoding on every batch-24 candidate, without training."""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up,encode
from feature263_signfix import encode_signfixed
from g1_joint_limit_gate import check_joint_limits,read_joint_limits
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_gpu_batch24_20261101';FIXED=ROOT/'omni_batch24_projection_20261101';HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=json.loads((SOURCE/'report.json').read_text());repairs=json.loads((FIXED/'report.json').read_text())
    assert sha(SOURCE/'records.jsonl')==source['records_sha256'] and sha(SOURCE/'report.json')==repairs['source_report_sha256']
    records=[json.loads(x) for x in (SOURCE/'records.jsonl').read_text().splitlines()]
    projected={r['sample_id']:r for r in repairs['cases'] if r['action']=='bounded_projection_preview'}
    limits=read_joint_limits(URDF);results=[]
    with (HERE/'rows.jsonl').open('x',buffering=1) as log:
        for r in records:
            try:
                sid=r['sample_id'];repair=projected.get(sid)
                state=(FIXED/(sid+'_state.npz')) if repair else (SOURCE/r['state20'])
                expected=repair['state_sha256'] if repair else r['state20_sha256']
                if sha(state)!=expected:raise ValueError('state hash mismatch')
                with np.load(state,allow_pickle=False) as d:
                    root=d['root_wxyz_xyz'];joints=d['joint_pos'];names=list(d['joint_names'])
                from g1_fk import URDFFK
                points=URDFFK(URDF).forward(root[:,4:],root[:,[1,2,3,0]],joints,names)
                raw=encode(points,foot_velocity_squared_threshold=.002)
                fixed=encode_signfixed(points,foot_velocity_squared_threshold=.002)
                if not np.array_equal(raw[:,1:],fixed[:,1:]):raise ValueError('non-root channel changed')
                with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(fixed),22).numpy()
                error=float(np.linalg.norm(canonicalize_z_up(points)[:-1]-decoded,axis=-1).max())
                limit=check_joint_limits(joints,names,limits,tolerance_rad=1e-6)
                if error>1e-4 or not limit['joint_limits_passed']:raise ValueError(f'technical state/feature check failed: {error}')
                changed=np.flatnonzero(raw[:,0]!=fixed[:,0]); output=None
                if len(changed):
                    out=HERE/(sid+'_signfixed.npz')
                    with out.open('xb') as f:np.savez_compressed(f,features=fixed)
                    output={'path':out.name,'sha256':sha(out)}
                result={'sample_id':sid,'status':'technical_checks_passed','state_sha256':expected,
                    'state_variant':'projected' if repair else 'original','angular_rows_changed':changed.tolist(),
                    'signfixed_feature_artifact':output,'max_decode_error_m':error,'training_ready':False}
            except Exception as e:
                result={'sample_id':r['sample_id'],'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            results.append(result);log.write(json.dumps(result,allow_nan=False)+'\n');log.flush()
    report={'source_report_sha256':sha(SOURCE/'report.json'),'projection_report_sha256':sha(FIXED/'report.json'),
        'module_sha256':sha(HERE/'feature263_signfix.py'),'script_sha256':sha(Path(__file__)),'rows_sha256':sha(HERE/'rows.jsonl'),
        'cases':len(results),'technical_passed':sum(x['status']=='technical_checks_passed' for x in results),
        'features_changed':[x['sample_id'] for x in results if x.get('angular_rows_changed')],
        'failed':[x['sample_id'] for x in results if x['status']=='failed'],'training_ready':False,
        'limitations':['Same FK and decoder; alternative angular channel is an experimental approximation, not author v2.',
            'No text semantic, source identity or license, ground/contact/collision, dynamics or training acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
