"""CPU-only fixed-window PyRoki scalability/overlap audit, not training data.

Input is original-rate, source-hashed GRAB SMPL-X joint positions. Overlapping
windows are saved separately so disagreement remains visible; no unsafe stitch,
resampling, automatic acceptance or modification of sources is performed.
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
VARIANT=ROOT/'pyroki_grab_temporal_20261030'
sys.path.insert(0,str(VARIANT))
sys.path.insert(1,str(ROOT/'foot_contact_fit_20261030'))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,create_conn_tree,solve_retargeting,foot_detect
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if any(x.platform!='cpu' for x in jax.devices()):raise ValueError('CPU required')
    source=ROOT/'grab_human_robot_pilot_20261030/pilot_arrays.npz'
    source_receipt=json.loads((source.parent/'report.json').read_text())
    variant_receipt=json.loads((VARIANT/'report.json').read_text())
    helper=VARIANT/'retarget_helpers/_utils.py'
    if sha(source)!=source_receipt['artifact_sha256'] or sha(source)!=variant_receipt['source_sha256']:
        raise ValueError('source provenance mismatch')
    if sha(helper)!=variant_receipt['helper_sha256']:raise ValueError('helper hash mismatch')
    with np.load(source,allow_pickle=False) as d:
        human=d['human_joints'].astype(np.float32);frames=d['input_frame_indices'].copy();fps=float(d['input_fps'])
    if human.shape!=(120,22,3) or fps!=120 or not np.all(np.diff(frames)==1):raise ValueError('unexpected timeline')
    urdf_path=PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if sha(urdf_path)!=variant_receipt['urdf_sha256']:raise ValueError('URDF changed')
    urdf=yourdfpy.URDF.load(str(urdf_path),load_meshes=False,load_collision_meshes=True)
    robot=pk.Robot.from_urdf(urdf);collision=pk.collision.RobotCollision.from_urdf(urdf)
    hi,ri=get_humanoid_retarget_indices_g1_27dof();mask=create_conn_tree(robot,ri)
    plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),
                                 height_data=jnp.zeros((100,100),dtype=jnp.float32))
    weights=variant_receipt['weights']; names=list(robot.joints.actuated_names)
    fk=URDFFK(urdf_path);offs,radii=foot_spheres(urdf_path)
    starts=list(range(1,113,4)); starts.append(112)
    if starts[-2]>=starts[-1] or len(set(starts))!=len(starts):raise ValueError('invalid schedule')
    previous=None;rows=[]; t0=time.monotonic()
    log=Path(__file__).with_name('windows.jsonl')
    with log.open('x',encoding='utf8',buffering=1) as stream:
        for wi,start in enumerate(starts):
            w=human[start:start+8];ind=frames[start+1:start+8]
            if w.shape!=(8,22,3):raise ValueError('bad window')
            left,right=foot_detect(w[None],thres=(.05/fps)**2)
            l=jnp.asarray(left[0]>0);r=jnp.asarray(right[0]>0);target=jnp.asarray(w[1:])
            footl=plane.project_points(target[:,10,:]);footr=plane.project_points(target[:,11,:])
            began=time.monotonic()
            root,joints=solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,
                is_left_foot_contact=l,is_right_foot_contact=r,left_foot_keypoints=footl,right_foot_keypoints=footr,
                smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,
                weights=weights,pick_point=jnp.zeros(3))
            q=np.asarray(joints);rq=np.asarray(root.wxyz_xyz)
            if q.shape!=(7,27) or rq.shape!=(7,7) or not np.isfinite(q).all() or not np.isfinite(rq).all():
                raise ValueError('nonfinite or malformed solver output')
            quats=rq[:,[1,2,3,0]]
            if not np.allclose(np.linalg.norm(quats,axis=1),1.,atol=1e-4,rtol=0):raise ValueError('nonunit root')
            centers=sphere_centers(fk,rq[:,4:],quats,q,names,offs)
            clearance=centers[...,2]-radii[None,:,:]
            speed=np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*fps
            overlap=None
            if previous is not None:
                common=np.intersect1d(previous['frames'],ind)
                old=np.searchsorted(previous['frames'],common);new=np.searchsorted(ind,common)
                quat0=Rotation.from_quat(previous['root'][old][:,[1,2,3,0]])
                quat1=Rotation.from_quat(rq[new][:,[1,2,3,0]])
                overlap={'frames':common.tolist(),
                    'max_root_position_disagreement_m':float(np.linalg.norm(previous['root'][old,4:]-rq[new,4:],axis=-1).max()),
                    'max_root_rotation_disagreement_rad':float((quat0.inv()*quat1).magnitude().max()),
                    'max_joint_disagreement_rad':float(np.abs(previous['joints'][old]-q[new]).max())}
            out=Path(__file__).with_name(f'window_{wi:02d}.npz')
            with out.open('xb') as f:np.savez_compressed(f,frames=ind,fps=np.array(fps),root_wxyz_xyz=rq,joint_pos=q,joint_names=np.asarray(names))
            row={'window':wi,'frames':ind.tolist(),'artifact':out.name,'artifact_sha256':sha(out),
                 'elapsed_s':time.monotonic()-began,'foot_min_clearance_assumed_z0_m':float(clearance.min()),
                 'max_foot_horizontal_speed_mps':float(speed.max()),'overlap_with_prior':overlap}
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            print('WINDOW',wi,'of',len(starts),'elapsed',round(row['elapsed_s'],2),flush=True)
            previous={'frames':ind,'root':rq,'joints':q};rows.append(row)
    report={'source_sha256':sha(source),'urdf_sha256':sha(urdf_path),'helper_sha256':sha(helper),
            'script_sha256':sha(__file__),'weights':weights,'fps':fps,'window_count':len(rows),
            'first_frame':int(rows[0]['frames'][0]),'last_frame':int(rows[-1]['frames'][-1]),
            'elapsed_s':time.monotonic()-t0,'rows_sha256':sha(log),
            'max_overlap_root_disagreement_m':max(r['overlap_with_prior']['max_root_position_disagreement_m'] for r in rows[1:]),
            'max_overlap_rotation_disagreement_rad':max(r['overlap_with_prior']['max_root_rotation_disagreement_rad'] for r in rows[1:]),
            'max_overlap_joint_disagreement_rad':max(r['overlap_with_prior']['max_joint_disagreement_rad'] for r in rows[1:]),
            'warning':'One subject and one short source action; 29 overlapping seven-frame windows, source floor z=0 unverified, v1 PyRoki weights/contacts are experimental. Window outputs not stitched or low-pass resampled; no object, learned policy, full dynamics or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
