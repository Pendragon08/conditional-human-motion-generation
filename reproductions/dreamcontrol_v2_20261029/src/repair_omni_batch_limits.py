"""Explore bounded limit projection of remaining Omni robot states.

The 0.03 rad exploration bound was selected after observing the current 12
failures. It is not a paper threshold or physical-quality acceptance. Preserve
all source states and captions; all outputs remain quarantined for review.
"""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
from feature263 import encode
from feature263_signfix import encode_signfixed
from g1_fk import URDFFK
from g1_joint_limit_gate import read_joint_limits,check_joint_limits
from audit_robot_states import audit_robot_states
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_gpu_remaining_20261101';HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=json.loads((SOURCE/'report.json').read_text());audit=json.loads((SOURCE/'audit_v2.json').read_text())
    if sha(SOURCE/'report.json')!=audit['report_sha256'] or sha(SOURCE/'records.jsonl')!=source['records_sha256'] or sha(SOURCE/'audit_rows_v2.jsonl')!=audit['audit_rows_sha256']:
        raise ValueError('source audit or records changed')
    records={r['sample_id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    failures=[r for r in map(json.loads,(SOURCE/'audit_rows_v2.jsonl').read_text().splitlines()) if r['status']=='failed']
    candidates=[r for r in failures if records[r['sample_id']]['joint_limits']['max_excess_rad']>1e-6]
    fk=URDFFK(URDF);limits=read_joint_limits(URDF);results=[]
    with (HERE/'records.jsonl').open('x',buffering=1) as log:
        for x in candidates:
            sid=x['sample_id'];r=records[sid]
            try:
                source_state=SOURCE/r['state20'];source_feature=SOURCE/r['feature']
                if sha(source_state)!=r['state20_sha256'] or sha(source_feature)!=r['feature_sha256']:raise ValueError('source changed')
                with np.load(source_state,allow_pickle=False) as d:
                    root=d['root_wxyz_xyz'];joint=d['joint_pos'];names=list(d['joint_names']);times=d['sample_times']
                    obj=d['object_wxyz_xyz'];frames=d['source_frame_positions']
                with np.load(source_feature,allow_pickle=False) as d:old=d['features']
                before=check_joint_limits(joint,names,limits,tolerance_rad=0.)
                if before['max_excess_rad']>.03:raise ValueError('outside post-hoc exploration bound')
                low=np.array([limits[n][0] for n in names]);high=np.array([limits[n][1] for n in names])
                fixed=np.clip(joint,low,high);points0=fk.forward(root[:,4:],root[:,[1,2,3,0]],joint,names)
                if not np.array_equal(encode(points0,foot_velocity_squared_threshold=.002),old):raise ValueError('source encoder changed')
                points=fk.forward(root[:,4:],root[:,[1,2,3,0]],fixed,names)
                features=encode_signfixed(points,foot_velocity_squared_threshold=.002)
                state=HERE/(sid+'_state.npz');feat=HERE/(sid+'_features.npz')
                with state.open('xb') as f:np.savez_compressed(f,root_wxyz_xyz=root,joint_pos=fixed,joint_names=np.asarray(names),
                    sample_times=times,object_wxyz_xyz=obj,source_frame_positions=frames)
                with feat.open('xb') as f:np.savez_compressed(f,features=features)
                evidence={'sample_id':sid,'state_path':state.name,'state_sha256':sha(state),'feature_path':feat.name,
                    'feature_sha256':sha(feat),'foot_velocity_squared_threshold':.002}
                check=audit_robot_states([evidence],feature_root=HERE,state_root=HERE,urdf_path=URDF,
                    tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5,reencode=encode_signfixed)
                if not check['state_technical_checks_passed']:raise ValueError('state preflight failed')
                delta=fixed-joint
                row={'sample_id':sid,'status':'technical_preview','parent_state_sha256':sha(source_state),
                    'parent_feature_sha256':sha(source_feature),'state':state.name,'state_sha256':sha(state),
                    'feature':feat.name,'feature_sha256':sha(feat),'joint_values_changed':int(np.count_nonzero(delta)),
                    'max_joint_correction_rad':float(abs(delta).max()),
                    'max_point_displacement_m':float(np.linalg.norm(points-points0,axis=-1).max()),
                    'max_joint_velocity_change_rad_s':float(np.max(abs(np.diff(delta,axis=0)))/.05),
                    'max_feature_difference':float(np.max(abs(features-old))),
                    'foot_proxy_values_changed':int(np.count_nonzero(features[:,-4:]!=old[:,-4:])),
                    'state_check':check['cases'][0],'caption_accepted':False,'training_ready':False}
            except Exception as e:
                row={'sample_id':sid,'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            results.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    result={'source_report_sha256':sha(SOURCE/'report.json'),'source_audit_sha256':sha(SOURCE/'audit_v2.json'),
        'script_sha256':sha(Path(__file__)),'module_sha256':sha(HERE/'feature263_signfix.py'),'rows_sha256':sha(HERE/'records.jsonl'),
        'candidate_count':len(candidates),'technical_previews':sum(x['status']=='technical_preview' for x in results),
        'failed':[x['sample_id'] for x in results if x['status']=='failed'],
        'post_hoc_exploration_bound_rad':.03,'training_ready':False,
        'limitations':['Bound was chosen after observing the data, not a paper filtering or physical-quality standard.',
        'The projection and optional root sign correction are approximations; FK and encoder are shared.',
        'No source/license/text/clock/contact/collision/dynamics or overall training acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
