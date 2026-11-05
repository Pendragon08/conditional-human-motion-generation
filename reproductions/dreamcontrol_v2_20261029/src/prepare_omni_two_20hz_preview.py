"""Isolated 20 Hz and 263D preview from two audited robot-to-robot fits.

Offline filtering and the 263D encoding are experimental; no training acceptance.
"""
import hashlib, json
from pathlib import Path
import numpy as np
from feature263 import encode
from g1_fk import URDFFK
from g1_joint_limit_gate import check_joint_limits, read_joint_limits
from resample_antialias_omniretarget import resample_antialias

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'omni_text_pair_fullrate_27dof_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    source_report=json.loads((SOURCE/'report.json').read_text())
    audit=json.loads((SOURCE/'audit.json').read_text())
    assert audit['source_report_sha256']==sha(SOURCE/'report.json')
    assert sha(URDF)==source_report['urdf27_sha256']
    results=[]; fk=URDFFK(URDF); limits=read_joint_limits(URDF)
    for case in source_report['cases']:
        src=SOURCE/case['output']; assert sha(src)==case['output_sha256']
        with np.load(src,allow_pickle=False) as z:
            root=z['root_wxyz_xyz']; joints=z['joint_pos']; names=z['joint_names']; obj=z['object_wxyz_xyz']
        q=np.concatenate([root,joints,obj],axis=1)
        resampled=resample_antialias(q,30.,20.)
        t=np.arange(len(resampled),dtype=np.float64)/20.
        assert t[-1] <= (len(q)-1)/30.
        xyz=resampled[:,4:7]; quat=resampled[:,[1,2,3,0]]; angles=resampled[:,7:34]
        points=fk.forward(xyz,quat,angles,list(names))
        features=encode(points,foot_velocity_squared_threshold=.002)
        check=check_joint_limits(angles,list(names),limits,tolerance_rad=1e-10)
        sid=case['sample_id']; state=HERE/(sid+'_state20.npz'); feat=HERE/(sid+'_263.npy')
        with state.open('xb') as f:
            np.savez_compressed(f,root_wxyz_xyz=resampled[:,:7],joint_pos=angles,joint_names=names,
                object_wxyz_xyz=resampled[:,34:41],points_z_up=points,sample_times=t,
                source_frame_positions=t*30.)
        with feat.open('xb') as f: np.save(f,features,allow_pickle=False)
        results.append({'sample_id':sid,'source_state_sha256':sha(src),'state':state.name,'state_sha256':sha(state),
            'features':feat.name,'features_sha256':sha(feat),'sampled_points':len(t),'feature_rows':len(features),
            'last_target_s':float(t[-1]),'last_source_s':float((len(q)-1)/30.),
            'discarded_tail_s':float((len(q)-1)/30.-t[-1]),
            'object_translation_last_sample_to_original_end_m':float(np.linalg.norm(resampled[-1,38:41]-obj[-1,4:7])),
            'joint_limit':check,'caption':case['source_caption'],'caption_semantics_accepted':False,'training_ready':False})
    report={'source_report_sha256':sha(SOURCE/'report.json'),'source_audit_sha256':sha(SOURCE/'audit.json'),
        'script_sha256':sha(Path(__file__)),'resampler_sha256':sha(HERE/'resample_antialias_omniretarget.py'),
        'urdf_sha256':sha(URDF),'source_fps_declared':30.,'target_fps_nominal':20.,
        'foot_velocity_squared_threshold':.002,'cases':results,'training_ready':False,
        'limitations':['Offline component-wise quaternion FIR and experimental encoder, not author v2.',
            'Feature contact channels use velocity proxy; ground, dynamics and timing not verified.',
            'Text has no per-frame labels; license, subject independence and semantics remain unaccepted.']}
    with (HERE/'report.json').open('x') as f: json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
