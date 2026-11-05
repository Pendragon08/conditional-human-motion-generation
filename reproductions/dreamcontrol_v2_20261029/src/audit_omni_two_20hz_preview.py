"""Audit saved 20 Hz previews; filtering is not independently recomputed."""
import hashlib, json
from pathlib import Path
import numpy as np
from feature263 import encode
from g1_fk import URDFFK
from g1_joint_limit_gate import check_joint_limits,read_joint_limits
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'omni_text_pair_fullrate_27dof_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    r=json.loads((HERE/'report.json').read_text()); src=json.loads((SOURCE/'report.json').read_text())
    assert sha(SOURCE/'report.json')==r['source_report_sha256']
    assert sha(SOURCE/'audit.json')==r['source_audit_sha256']
    assert sha(URDF)==r['urdf_sha256']
    cases=[]
    for c in r['cases']:
        sc=next(x for x in src['cases'] if x['sample_id']==c['sample_id'])
        assert sha(SOURCE/sc['output'])==c['source_state_sha256']
        assert sha(HERE/c['state'])==c['state_sha256'] and sha(HERE/c['features'])==c['features_sha256']
        with np.load(HERE/c['state'],allow_pickle=False) as d:
            times=d['sample_times']; frames=d['source_frame_positions']; root=d['root_wxyz_xyz']
            joint=d['joint_pos']; names=list(d['joint_names']); points=d['points_z_up']; obj=d['object_wxyz_xyz']
        assert np.array_equal(times,np.arange(len(times))/20.) and np.array_equal(frames,times*30.)
        assert np.isfinite(obj).all() and np.max(abs(np.linalg.norm(obj[:,:4],axis=1)-1))<1e-10
        limits=check_joint_limits(joint,names,read_joint_limits(URDF),tolerance_rad=1e-10)
        assert limits['joint_limits_passed']
        reconstructed=URDFFK(URDF).forward(root[:,4:7],root[:,[1,2,3,0]],joint,names)
        assert np.max(abs(reconstructed-points))<1e-10
        expected=np.load(HERE/c['features'],allow_pickle=False)
        generated=encode(reconstructed,foot_velocity_squared_threshold=.002)
        assert expected.shape==(len(times)-1,263) and np.array_equal(expected,generated)
        assert len(times)==c['sampled_points'] and len(expected)==c['feature_rows']
        cases.append({'sample_id':c['sample_id'],'samples':len(times),'feature_rows':len(expected),
            'full_reencoding_identical':True,'joint_limits_passed':True})
    out={'source_report_sha256':sha(HERE/'report.json'),'audit_script_sha256':sha(Path(__file__)),
        'cases':cases,'limitations':['Filter not independently recomputed; FK and encoder reuse original modules.',
        'No source identity, time truth, text semantics, license or physics acceptance.'],'training_ready':False}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
