"""Compare five overlapping anchored windows with audited S3 baseline.

This is an experimental soft constraint using the prior solution, not the
unpublished author-v2 optimization or a training dataset.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(1,str(ROOT/'foot_contact_fit_20261030'))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,create_conn_tree,solve_retargeting,foot_detect
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers
from waist_lock_ik import FIT_LINKS

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if any(x.platform!='cpu' for x in jax.devices()):raise ValueError('CPU required')
    base=ROOT/'pyroki_grab_s3_windowed_20261030';source=ROOT/'grab_s3_human_source_20261030/human_source.npz'
    baseline=json.loads((base/'report.json').read_text()); audit=json.loads((base/'audit_report.json').read_text())
    manifest=json.loads(Path(__file__).with_name('manifest.json').read_text());helper=Path(__file__).with_name('retarget_helpers')/'_utils.py'
    if sha(base/'report.json')!=audit['report_sha256'] or sha(base/'windows.jsonl')!=baseline['rows_sha256'] or sha(source)!=baseline['source_sha256']:
        raise ValueError('input provenance mismatch')
    if sha(helper)!=manifest['anchor_helper_sha256']:raise ValueError('helper hash mismatch')
    rows=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    if len(rows)!=29 or audit['windows_verified']!=29:raise ValueError('baseline incomplete')
    with np.load(source,allow_pickle=False) as d:
        human=d['human_joints'].astype(np.float32);source_frames=d['input_frame_indices'].copy();fps=float(d['input_fps'])
    if human.shape!=(120,22,3) or fps!=120.:raise ValueError('unexpected source')
    first=base/rows[0]['artifact']
    if sha(first)!=rows[0]['artifact_sha256']:raise ValueError('seed mismatch')
    with np.load(first,allow_pickle=False) as d:
        prior={'frames':d['frames'].copy(),'root':d['root_wxyz_xyz'].copy(),'joint':d['joint_pos'].copy()}
    urdf_path=PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if sha(urdf_path)!=baseline['urdf_sha256']:raise ValueError('URDF mismatch')
    urdf=yourdfpy.URDF.load(str(urdf_path),load_meshes=False,load_collision_meshes=True)
    robot=pk.Robot.from_urdf(urdf);collision=pk.collision.RobotCollision.from_urdf(urdf)
    hi,ri=get_humanoid_retarget_indices_g1_27dof();mask=create_conn_tree(robot,ri)
    plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),height_data=jnp.zeros((100,100),dtype=jnp.float32))
    fk=URDFFK(urdf_path);offs,radii=foot_spheres(urdf_path);names=list(robot.joints.actuated_names)
    weight=10.;records=[]
    with Path(__file__).with_name('windows.jsonl').open('x',encoding='utf8',buffering=1) as stream:
        for i in range(1,6):
            row=rows[i]; expected=np.asarray(row['frames']); source_start=int(expected[0]-source_frames[0]-1)
            raw=human[source_start:source_start+8]
            if raw.shape!=(8,22,3) or not np.array_equal(source_frames[source_start+1:source_start+8],expected):raise ValueError('timeline mismatch')
            common=np.intersect1d(prior['frames'],expected); old=np.searchsorted(prior['frames'],common);new=np.searchsorted(expected,common)
            if len(common)!=3 or not np.array_equal(new,np.arange(3)):raise ValueError('unexpected overlap')
            roots=np.zeros((7,7),dtype=np.float32);roots[:,0]=1.;qs=np.zeros((7,27),dtype=np.float32);active=np.zeros(7,dtype=np.float32)
            roots[new]=prior['root'][old];qs[new]=prior['joint'][old];active[new]=1.
            left,right=foot_detect(raw[None],thres=(.05/fps)**2);target=jnp.asarray(raw[1:])
            began=time.monotonic()
            root,joints=solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,
                is_left_foot_contact=jnp.asarray(left[0]>0),is_right_foot_contact=jnp.asarray(right[0]>0),
                left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),
                smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,
                weights=baseline['weights'],pick_point=jnp.zeros(3),anchor_roots=jnp.asarray(roots),
                anchor_joints=jnp.asarray(qs),anchor_mask=jnp.asarray(active),anchor_weight=weight)
            root=np.asarray(root.wxyz_xyz);q=np.asarray(joints)
            if root.shape!=(7,7) or q.shape!=(7,27) or not np.isfinite(root).all() or not np.isfinite(q).all():raise ValueError('invalid output')
            q0=Rotation.from_quat(prior['root'][old][:,[1,2,3,0]]);q1=Rotation.from_quat(root[new][:,[1,2,3,0]])
            mismatch={'max_root_position_m':float(np.linalg.norm(prior['root'][old,4:]-root[new,4:],axis=1).max()),
                      'max_root_angle_rad':float((q0.inv()*q1).magnitude().max()),
                      'max_joint_angle_rad':float(np.abs(prior['joint'][old]-q[new]).max())}
            centers=sphere_centers(fk,root[:,4:],root[:,[1,2,3,0]],q,names,offs)
            speed=np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*fps
            points=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=FIT_LINKS)
            human_error=np.linalg.norm(points-raw[1:,[1,2,4,5,7,8,16,17,18,19,20,21]],axis=-1)
            out=Path(__file__).with_name(f'window_{i:02d}.npz')
            with out.open('xb') as f:np.savez_compressed(f,frames=expected,root_wxyz_xyz=root,joint_pos=q,joint_names=np.asarray(names))
            record={'window':i,'frames':expected.tolist(),'artifact_sha256':sha(out),'elapsed_s':time.monotonic()-began,
                    'overlap':mismatch,'baseline_overlap':row['overlap_with_prior'],
                    'foot_min_clearance_m':float((centers[...,2]-radii[None,:,:]).min()),
                    'max_foot_horizontal_speed_mps':float(speed.max()),
                    'unscaled_world_keypoint_mean_m':float(human_error.mean())}
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            print('WINDOW',i,'done',flush=True);prior={'frames':expected,'root':root,'joint':q};records.append(record)
    report={'source_sha256':sha(source),'baseline_report_sha256':sha(base/'report.json'),
            'seed_artifact_sha256':sha(first),'urdf_sha256':sha(urdf_path),'helper_sha256':sha(helper),
            'script_sha256':sha(__file__),'rows_sha256':sha(Path(__file__).with_name('windows.jsonl')),
            'anchor_weight':weight,'windows':len(records),
            'max_overlap_root_m':max(x['overlap']['max_root_position_m'] for x in records),
            'max_baseline_overlap_root_m':max(x['baseline_overlap']['max_root_position_disagreement_m'] for x in records),
            'warning':'Five windows of one S3 action, initialized by the previously computed first window; soft anchor strength and v1 weights are experimental. Ground z=0, low-speed foot proxies, no object/dynamics; human absolute world-point error ignores morphology. Not a paper method, physical validation or training data.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
