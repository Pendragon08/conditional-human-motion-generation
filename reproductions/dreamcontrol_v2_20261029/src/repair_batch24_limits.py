"""Isolated, bounded numerical limit projection for batch-24 failures.

Maximum 0.001 rad is an explicit engineering bound, not a paper threshold.
The original candidate and all upstream files are preserved. Does not accept
text, original timing, contact/collision, dynamics, or training quality.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up,encode
from g1_fk import URDFFK
from g1_joint_limit_gate import check_joint_limits,read_joint_limits
from audit_robot_states import audit_robot_states
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent; SOURCE=ROOT/'omni_gpu_batch24_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
NAMES=('sub10_largebox_041','sub10_largebox_014','sub16_largebox_031','sub8_largebox_006')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    source_report=json.loads((SOURCE/'report.json').read_text()); audit=json.loads((SOURCE/'audit_v2.json').read_text())
    assert sha(SOURCE/'report.json')==audit['report_sha256'] and sha(SOURCE/'records.jsonl')==source_report['records_sha256']
    assert set(audit['failed'])==set(NAMES) and sha(URDF)==source_report['urdf27_sha256']
    records={r['sample_id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    fk=URDFFK(URDF);limits=read_joint_limits(URDF);results=[]
    for sid in NAMES:
        r=records[sid];state=SOURCE/r['state20'];feature=SOURCE/r['feature']
        assert sha(state)==r['state20_sha256'] and sha(feature)==r['feature_sha256']
        with np.load(state,allow_pickle=False) as f:
            root=f['root_wxyz_xyz'];q=f['joint_pos'];names=list(f['joint_names']);times=f['sample_times']
            obj=f['object_wxyz_xyz'];frames=f['source_frame_positions']
        with np.load(feature,allow_pickle=False) as f:old=f['features']
        before=check_joint_limits(q,names,limits,tolerance_rad=0.)
        original=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names)
        with torch.no_grad(): decoded=recover_from_ric(torch.from_numpy(old),22).numpy()
        errors=np.linalg.norm(canonicalize_z_up(original)[:-1]-decoded,axis=-1)
        worst=np.unravel_index(errors.argmax(),errors.shape)
        if before['max_excess_rad']==0:
            results.append({'sample_id':sid,'action':'quarantine_unresolved_fk_decode',
                'maximum_fk_decode_error_m':float(errors.max()),'worst_feature_row':int(worst[0]),
                'worst_keypoint':int(worst[1]),'training_ready':False})
            continue
        if before['max_excess_rad']>.001:raise ValueError('outside explicit projection bound')
        lo=np.array([limits[x][0] for x in names]);hi=np.array([limits[x][1] for x in names])
        fixed=np.clip(q,lo,hi);points=fk.forward(root[:,4:],root[:,[1,2,3,0]],fixed,names)
        features=encode(points,foot_velocity_squared_threshold=.002); state_out=HERE/(sid+'_state.npz');feat_out=HERE/(sid+'_features.npz')
        with state_out.open('xb') as f:np.savez_compressed(f,root_wxyz_xyz=root,joint_pos=fixed,joint_names=np.asarray(names),
            sample_times=times,object_wxyz_xyz=obj,source_frame_positions=frames)
        with feat_out.open('xb') as f:np.savez_compressed(f,features=features)
        review=audit_robot_states([{'sample_id':sid,'state_path':state_out.name,'state_sha256':sha(state_out),
            'feature_path':feat_out.name,'feature_sha256':sha(feat_out),'foot_velocity_squared_threshold':.002}],
            feature_root=HERE,state_root=HERE,urdf_path=URDF,tolerance_rad=1e-6,
            fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
        results.append({'sample_id':sid,'action':'bounded_projection_preview','parent_state_sha256':sha(state),
            'parent_feature_sha256':sha(feature),'state_sha256':sha(state_out),'feature_sha256':sha(feat_out),
            'values_changed':int(np.count_nonzero(fixed-q)),'max_correction_rad':float(np.max(abs(fixed-q))),
            'max_point_displacement_m':float(np.linalg.norm(points-original,axis=-1).max()),
            'max_feature_difference':float(np.max(abs(features-old))),
            'foot_proxy_values_changed':int(np.count_nonzero(features[:,-4:]!=old[:,-4:])),
            'technical_check':review,'training_ready':False})
    result={'source_report_sha256':sha(SOURCE/'report.json'),'source_audit_sha256':sha(SOURCE/'audit_v2.json'),
        'script_sha256':sha(Path(__file__)),'projection_bound_rad':.001,'cases':results,'training_ready':False,
        'limitations':['Numerical projection is an engineering approximation, not author-v2 filter or physics.',
                       'No semantic, license, source clock, contact/collision or dynamics acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
