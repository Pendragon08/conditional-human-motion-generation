"""Compare matched S1 windows; diagnostic only, not physical acceptance."""
import hashlib,json,sys
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from mujoco_contact_flags import flags

LINKS=('pelvis_contour_link','left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
       'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_roll_link','right_shoulder_roll_link',
       'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link')
INDICES=(0,1,2,4,5,7,8,16,17,18,19,20,21)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    old=ROOT/'pyroki_grab_s1_pour_windowed_20261030';new=ROOT/'pyroki_s1_sole_pilot_20261030';src=ROOT/'grab_s1_pour_human_source_20261030/human_source.npz'
    receipts=[json.loads((p/'report.json').read_text()) for p in (old,new)];audits=[json.loads((p/'audit_report.json').read_text()) for p in (old,new)]
    for p,r,a in zip((old,new),receipts,audits):
        if sha(p/'report.json')!=a['report_sha256'] or sha(p/'windows.jsonl')!=r['rows_sha256'] or sha(src)!=r['source_sha256'] or sha(src)!=a['source_sha256']:raise ValueError('provenance mismatch')
    if audits[0]['windows_verified']!=29 or audits[1]['windows_verified']!=5:raise ValueError('wrong window count')
    bw=dict(receipts[1]['weights']);sole=bw.pop('sole_contact')
    if bw!=receipts[0]['weights'] or sole!=10.:raise ValueError('unexpected weight differences')
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    if any(sha(urdf)!=x['urdf_sha256'] for x in receipts):raise ValueError('URDF changed')
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
    model=mujoco.MjModel.from_xml_path(str(scene));fk=URDFFK(urdf);offsets,radii=foot_spheres(urdf)
    with np.load(src,allow_pickle=False) as z:human=z['human_joints'].copy();source_frames=z['input_frame_indices'].copy();fps=float(z['input_fps'])
    if human.shape!=(120,22,3) or fps!=120:raise ValueError('bad source')
    old_rows=[json.loads(x) for x in (old/'windows.jsonl').read_text().splitlines()][:5];new_rows=[json.loads(x) for x in (new/'windows.jsonl').read_text().splitlines()]
    if len(old_rows)!=5 or len(new_rows)!=5:raise ValueError('wrong matched sample count')
    output=[]
    for i,(ar,br) in enumerate(zip(old_rows,new_rows)):
        if ar['window']!=i or br['window']!=i or ar['frames']!=br['frames']:raise ValueError('window mismatch')
        row={'window':i,'frames':ar['frames']}
        for label,directory,receipt in (('baseline',old,ar),('sole_variant',new,br)):
            path=directory/receipt['artifact']
            if sha(path)!=receipt['artifact_sha256']:raise ValueError('window changed')
            with np.load(path,allow_pickle=False) as z:
                frames=z['frames'].copy();roots=z['root_wxyz_xyz'].copy();joint=z['joint_pos'].copy();names=z['joint_names'].tolist()
            if len(frames)!=7 or not np.array_equal(frames,np.asarray(ar['frames'])):raise ValueError('wrong frames')
            ix=np.searchsorted(source_frames,frames)
            if not np.array_equal(source_frames[ix],frames):raise ValueError('misaligned source')
            points=fk.forward(roots[:,4:],roots[:,[1,2,3,0]],joint,names,links=LINKS)
            errors=np.linalg.norm(points-human[ix][:,INDICES,:],axis=2)
            centers=sphere_centers(fk,roots[:,4:],roots[:,[1,2,3,0]],joint,names,offsets)
            clearance=centers[...,2]-radii[None,:,:];foot_min=clearance.min(axis=2)
            vel=np.linalg.norm(np.diff(centers,axis=0)[...,:2]*fps,axis=-1)
            contacts=flags(model,{'root_pos':roots[:,4:],'root_quat_xyzw':roots[:,[1,2,3,0]],'joint_pos':joint,'joint_names':names})
            row[label]={'keypoint_world_error_mean_m':float(errors.mean()),'keypoint_world_error_max_m':float(errors.max()),
                        'foot_sphere_min_clearance_m':float(clearance.min()),'foot_sphere_max_penetration_m':float(np.maximum(-clearance,0).max()),
                        'foot_frame_min_clearance_mean_m':float(foot_min.mean()),'foot_frame_near_plane_fraction':float(np.mean(np.abs(foot_min)<=.005)),
                        'max_foot_horizontal_speed_mps':float(vel.max()),'mean_foot_horizontal_speed_mps':float(vel.mean()),
                        'mujoco_per_foot_contact_fraction':float(contacts.mean())}
        output.append(row)
    summary={}
    for label in ('baseline','sole_variant'):
        keys=output[0][label].keys()
        summary[label]={key:float(np.mean([row[label][key] for row in output])) for key in keys}
    result={'source_sha256':sha(src),'baseline_report_sha256':sha(old/'report.json'),'variant_report_sha256':sha(new/'report.json'),
            'variant_audit_sha256':sha(new/'audit_report.json'),'urdf_sha256':sha(urdf),'scene_sha256':sha(scene),'script_sha256':sha(Path(__file__)),
            'sole_weight':sole,'windows':output,'mean_of_window_metrics':summary,
            'warning':'Five overlapping windows from one S1 tuning clip, not independent holdout. Low-speed contact and z=0 are assumptions. Variant changes ankle-z loss to URDF sphere clearance and uses experimental weight; MuJoCo per-frame static contact is not real force or dynamics. Human/robot world keypoint distance is not scale-corrected. No paper/training acceptance.'}
    with Path(__file__).with_name('comparison.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
