"""Check the isolated GRAB pair, including the existing FK/state validator."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_robot_states import audit_robot_states

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
BLEND=ROOT/'pyroki_window_blend_v2_20261030/s3_blended.npz'
ANNOTATION=ROOT/'grab_s3_manual_caption_20261101/annotation.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=json.loads((HERE/'manifest.json').read_text());assert len(manifest['records'])==1
    rec=manifest['records'][0];annotation=json.loads(ANNOTATION.read_text())
    assert rec['caption']==annotation['caption'] and rec['manual_caption_receipt_sha256']==sha(ANNOTATION)
    assert rec['split']=='unassigned' and rec['training_ready'] is False and manifest['training_ready'] is False
    assert sha(BLEND)==manifest['parent_blend_sha256'] and sha(HERE/rec['feature_path'])==rec['feature_sha256']
    with np.load(BLEND,allow_pickle=False) as old,np.load(HERE/rec['state_path'],allow_pickle=False) as state:
        np.testing.assert_array_equal(state['root_wxyz_xyz'],old['qpos20'][:,:7])
        np.testing.assert_array_equal(state['joint_pos'],old['qpos20'][:,7:])
        np.testing.assert_array_equal(state['joint_names'],old['joint_names'])
        np.testing.assert_allclose(state['sample_times'],old['frames20']/120.,atol=0,rtol=0)
    result=audit_robot_states([rec],feature_root=HERE,state_root=HERE,urdf_path=URDF,
                              tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
    out={'manifest_sha256':sha(HERE/'manifest.json'),'annotation_sha256':sha(ANNOTATION),
         'script_sha256':sha(Path(__file__)),'state_audit':result,'training_ready':False,
         'limitations':['Technical state check uses same FK/encoder as export.',
                        'No independent source timing, manual semantic review, contact, dynamics or overall training acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
