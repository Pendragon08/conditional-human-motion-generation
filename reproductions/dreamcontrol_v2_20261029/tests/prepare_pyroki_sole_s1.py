"""Create an isolated URDF-foot-sphere contact loss experiment for S1.

This is an explicit approximation, not the unpublished author-v2 retargeter.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
VARIANT=ROOT/'pyroki_sole_variant_20261030'
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from foot_geometry_audit import foot_spheres

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def replace_one(text,old,new):
    if text.count(old)!=1:raise ValueError('ambiguous edit: '+old[:60])
    return text.replace(old,new)
def write(path,text):
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)

def main():
    original=ROOT/'pyroki_grab_temporal_20261030';prev=json.loads((original/'report.json').read_text());helper=original/'retarget_helpers/_utils.py'
    if sha(helper)!=prev['helper_sha256']:raise ValueError('legacy helper changed')
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    if sha(urdf)!=prev['urdf_sha256']:raise ValueError('URDF changed')
    offsets,radii=foot_spheres(urdf)
    if offsets.shape!=(2,4,3) or radii.shape!=(2,4) or not np.isfinite(offsets).all():raise ValueError('unexpected sole geometry')
    source=helper.read_text();marker='    # Variables.\n'
    insert=('    # Experimental local foot collision spheres; fixed from the verified URDF.\n'
            f'    sole_offsets = jnp.asarray({offsets.tolist()})\n'
            f'    sole_radii = jnp.asarray({radii.tolist()})\n\n')
    source=replace_one(source,marker,insert+marker)
    old='''        return (
            jnp.concatenate(
                [
                    left_foot_contact_cost.flatten(),
                    right_foot_contact_cost.flatten(),
                    left_foot_contact_residual_rot.flatten(),
                    right_foot_contact_residual_rot.flatten(),
                ]
            )
            * weights["floor_contact"]
        )'''
    new='''        # Keep XY ankle targets and orientation, but replace the old ankle-z
        # target with a plane clearance of the lowest URDF foot sphere.
        # This assumes a flat known plane and is not an author-v2 loss.
        left_centers = left_foot_pos + (left_foot_ori @ sole_offsets[0].T).T
        right_centers = right_foot_pos + (right_foot_ori @ sole_offsets[1].T).T
        left_clearance = jnp.min(left_centers[:, 2] - sole_radii[0]) - left_foot_keypoints[2]
        right_clearance = jnp.min(right_centers[:, 2] - sole_radii[1]) - right_foot_keypoints[2]
        base = jnp.concatenate([left_foot_contact_cost[:2], right_foot_contact_cost[:2],
                                left_foot_contact_residual_rot.flatten(), right_foot_contact_residual_rot.flatten()]) * weights["floor_contact"]
        sole = jnp.stack([is_left_foot_contact * left_clearance,
                          is_right_foot_contact * right_clearance]) * weights["sole_contact"]
        return jnp.concatenate([base, sole])'''
    source=replace_one(source,old,new)
    target=VARIANT/'retarget_helpers';target.mkdir(parents=True,exist_ok=False)
    write(target/'__init__.py',(original/'retarget_helpers/__init__.py').read_text())
    write(target/'_utils.py',source)
    weights=dict(prev['weights']);weights['sole_contact']=10.0
    variant_receipt={'helper_sha256':sha(target/'_utils.py'),'original_helper_sha256':sha(helper),'urdf_sha256':sha(urdf),'weights':weights,
                     'sole_offsets':offsets.tolist(),'sole_radii':radii.tolist(),
                     'warning':'Experimental 10x sole-clearance residual, old ankle-z residual removed, assumed z=0 and source low-speed mask. Not paper author method or physical acceptance.'}
    write(VARIANT/'report.json',json.dumps(variant_receipt,indent=2,allow_nan=False))
    base=ROOT/'pyroki_grab_s1_pour_windowed_20261030';base_report=json.loads((base/'report.json').read_text())
    driver=base/'pyroki_grab_windowed_pilot.py';audit=base/'audit_pyroki_windowed_pilot.py'
    if sha(driver)!=base_report['script_sha256']:raise ValueError('S1 baseline driver changed')
    code=replace_one(driver.read_text(),"VARIANT=ROOT/'pyroki_grab_temporal_20261030'","VARIANT=ROOT/'pyroki_sole_variant_20261030'")
    code=replace_one(code,'starts=list(range(1,113,4)); starts.append(112)','starts=list(range(1,21,4))')
    write(HERE/driver.name,code)
    code=audit.read_text()
    code=replace_one(code,"helper=ROOT/'pyroki_grab_temporal_20261030/retarget_helpers/_utils.py'","helper=ROOT/'pyroki_sole_variant_20261030/retarget_helpers/_utils.py'")
    code=replace_one(code,"if len(rows)!=29 or report['window_count']!=29:","if len(rows)!=5 or report['window_count']!=5:")
    code=replace_one(code,"list(range(int(source_frames[2]),int(source_frames[-1])+1))","list(range(int(source_frames[2]),int(rows[-1]['frames'][-1])+1))")
    write(HERE/audit.name,code)
    out={'baseline_driver_sha256':sha(driver),'baseline_audit_sha256':sha(audit),'variant_helper_sha256':sha(target/'_utils.py'),
         'variant_report_sha256':sha(VARIANT/'report.json'),'driver_sha256':sha(HERE/driver.name),'audit_sha256':sha(HERE/audit.name),
         'builder_sha256':sha(Path(__file__)),'windows':5,'sole_contact_weight':10.0,
         'limitations':'Hypothesis on tuning sample S1; flat plane and low-speed contact assumption. Old ankle-z loss replaced by URDF sphere clearance, not author v2 implementation or physical acceptance.'}
    write(HERE/'manifest.json',json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
