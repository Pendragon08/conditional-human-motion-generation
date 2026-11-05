"""Experimental legacy-PyRoki variant that evaluates global alignment after offset."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
BASE=ROOT/'pyroki_s4_offset_diagnostic_20261030'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def replace_once(s,old,new):
    if s.count(old)!=1:raise ValueError('unexpected target count')
    return s.replace(old,new)

def main():
    man=json.loads((BASE/'manifest.json').read_text());src=BASE/'retarget_helpers/_utils.py'
    if sha(src)!=man['instrumented_helper_sha256']:raise ValueError('instrumented base changed')
    s=src.read_text()
    s=replace_once(s,'        var_robot_cfg: jaxls.Var[jnp.ndarray],\n        keypoints: jnp.ndarray,\n    ) -> jax.Array:\n        """Soft cost to align the human keypoints to the robot, in the world frame."""',
        '        var_robot_cfg: jaxls.Var[jnp.ndarray],\n        var_offset: OffsetVar,\n        keypoints: jnp.ndarray,\n    ) -> jax.Array:\n        """Soft cost to align the final offset-adjusted robot in the world frame."""')
    s=replace_once(s,'        link_pos = T_world_link.translation()[g1_joint_retarget_indices]\n        keypoint_pos = keypoints[smpl_joint_retarget_indices]',
        '        link_pos = T_world_link.translation()[g1_joint_retarget_indices] + var_values[var_offset]\n        keypoint_pos = keypoints[smpl_joint_retarget_indices]')
    s=replace_once(s,'        pc_alignment_cost(\n            var_Ts_world_root,\n            var_joints,\n            target_keypoints,',
        '        pc_alignment_cost(\n            var_Ts_world_root,\n            var_joints,\n            var_offset,\n            target_keypoints,')
    pkg=OUT/'retarget_helpers';pkg.mkdir();(pkg/'__init__.py').open('x').close()
    with (pkg/'_utils.py').open('x',encoding='utf8',newline='\n') as f:f.write(s)
    base_driver=BASE/'pyroki_offset_diagnostic.py'
    if sha(base_driver)!=json.loads((BASE/'report.json').read_text())['driver_sha256']:raise ValueError('base driver changed')
    driver=base_driver.read_text()
    # Diagnostic driver authenticates the helper through this manifest key.
    with (OUT/'pyroki_offset_diagnostic.py').open('x',encoding='utf8',newline='\n') as f:f.write(driver)
    # The existing driver expects the unchanged original temporal report hash.
    result={'base_report_sha256':man['base_report_sha256'],'base_instrumented_helper_sha256':sha(src),
            'instrumented_helper_sha256':sha(pkg/'_utils.py'),'driver_sha256':sha(OUT/'pyroki_offset_diagnostic.py'),
            'script_sha256':sha(__file__),
            'changes':'Add shared offset variable to world-keypoint residual and its factor arguments; no other objective terms or weights changed.',
            'limitations':'Exploratory consistency variant of legacy code, assumed ground and one tuning window; not published v2 method or training acceptance.'}
    with (OUT/'manifest.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
