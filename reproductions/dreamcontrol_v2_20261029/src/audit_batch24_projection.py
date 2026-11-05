"""Independently recheck bounded limit projection; not physics acceptance."""
import hashlib,json
from pathlib import Path
import numpy as np
from g1_fk import URDFFK
from g1_joint_limit_gate import read_joint_limits
from feature263 import encode
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;SOURCE=ROOT/'omni_gpu_batch24_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    report=json.loads((HERE/'report.json').read_text());records={x['sample_id']:x for x in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    assert sha(SOURCE/'report.json')==report['source_report_sha256'] and sha(SOURCE/'audit_v2.json')==report['source_audit_sha256']
    results=[];limits=read_joint_limits(URDF);robot=URDFFK(URDF)
    for case in report['cases']:
        if case['action']!='bounded_projection_preview':continue
        sid=case['sample_id'];old=SOURCE/records[sid]['state20'];new=HERE/(sid+'_state.npz');feat=HERE/(sid+'_features.npz')
        assert sha(old)==case['parent_state_sha256'] and sha(new)==case['state_sha256'] and sha(feat)==case['feature_sha256']
        with np.load(old,allow_pickle=False) as a, np.load(new,allow_pickle=False) as b:
            before=a['joint_pos'];after=b['joint_pos'];names=list(a['joint_names'])
            for k in ('root_wxyz_xyz','sample_times','joint_names','object_wxyz_xyz','source_frame_positions'):
                np.testing.assert_array_equal(a[k],b[k])
            root=b['root_wxyz_xyz']
        expected=np.empty_like(before)
        for j,name in enumerate(names):
            lo,hi=limits[name]
            expected[:,j]=[min(max(float(v),lo),hi) for v in before[:,j]]
        np.testing.assert_array_equal(expected,after)
        delta=np.abs(after-before); assert abs(float(delta.max())-case['max_correction_rad'])<1e-15
        assert int(np.count_nonzero(delta))==case['values_changed'] and float(delta.max())<=report['projection_bound_rad']
        pts=robot.forward(root[:,4:],root[:,[1,2,3,0]],after,names)
        actual=encode(pts,foot_velocity_squared_threshold=.002)
        with np.load(feat,allow_pickle=False) as f:np.testing.assert_array_equal(actual,f['features'])
        results.append({'sample_id':sid,'max_correction_rad':float(delta.max()),'feature_identical':True,'training_ready':False})
    assert len(results)==3
    result={'report_sha256':sha(HERE/'report.json'),'script_sha256':sha(Path(__file__)),
        'cases':results,'training_ready':False,'limitations':['FK and encoder share original modules.',
        'Projection is numerical engineering, no real contact, semantic or license acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
