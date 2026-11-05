"""Soft-anchor consecutive 120-Hz ACCAD robot windows on CPU only.

The first solution and every input are hash-checked against an audited baseline.
Anchor weight is exploratory and does not certify caption or physical quality.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'accad_013267_pyroki_fullrate_20261031'
SOURCE=ROOT/'accad_013267_fullrate_20261031/human_120hz.npz'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
sys.path.insert(0,str(HERE))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,create_conn_tree,solve_retargeting,foot_detect


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    if any(d.platform!='cpu' for d in jax.devices()):raise ValueError('CPU required')
    manifest=json.loads((HERE/'manifest.json').read_text())
    baseline=json.loads((BASE/'report.json').read_text());base_audit=json.loads((BASE/'audit_report.json').read_text())
    helper=HERE/'retarget_helpers/_utils.py'
    if (sha(BASE/'report.json')!=base_audit['report_sha256'] or sha(BASE/'windows.jsonl')!=baseline['rows_sha256']
        or sha(SOURCE)!=baseline['source_sha256'] or sha(URDF)!=baseline['urdf_sha256']
        or sha(helper)!=manifest['helper_sha256'] or base_audit['windows_verified']!=88):
        raise ValueError('baseline provenance mismatch')
    rows=[json.loads(x) for x in (BASE/'windows.jsonl').read_text().splitlines()]
    with np.load(SOURCE,allow_pickle=False) as d:
        human=d['human'].astype(np.float32);frames=d['source_frames'].copy();times=d['source_times'].copy()
    if human.shape!=(355,22,3) or not np.array_equal(frames,np.arange(355)):
        raise ValueError('source timeline mismatch')
    urdf=yourdfpy.URDF.load(str(URDF),load_meshes=False,load_collision_meshes=True)
    robot=pk.Robot.from_urdf(urdf);collision=pk.collision.RobotCollision.from_urdf(urdf)
    hi,ri=get_humanoid_retarget_indices_g1_27dof();mask=create_conn_tree(robot,ri)
    plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),
                                 height_data=jnp.zeros((100,100),dtype=jnp.float32))
    names=list(robot.joints.actuated_names); records=[];previous=None; began=time.monotonic()
    with (HERE/'windows.jsonl').open('x',encoding='utf8',buffering=1) as stream:
        for wi,row in enumerate(rows):
            indices=np.asarray(row['sample_indices'],dtype=int)
            if len(indices)!=7 or not np.array_equal(indices,np.arange(indices[0],indices[0]+7)):
                raise ValueError('unexpected baseline indices')
            if wi==0:
                seed=BASE/row['artifact']
                if sha(seed)!=row['artifact_sha256']:raise ValueError('seed changed')
                with np.load(seed,allow_pickle=False) as d:
                    root=d['root_wxyz_xyz'].copy();q=d['joint_pos'].copy()
                    if [str(x) for x in d['joint_names']]!=names:raise ValueError('seed names changed')
            else:
                overlap=np.intersect1d(previous['indices'],indices)
                old=np.searchsorted(previous['indices'],overlap);new=np.searchsorted(indices,overlap)
                if len(overlap) not in (3,4) or not np.array_equal(new,np.arange(len(overlap))):
                    raise ValueError('unexpected overlap')
                anchor_root=np.zeros((7,7),np.float32);anchor_root[:,0]=1.
                anchor_q=np.zeros((7,27),np.float32);active=np.zeros(7,np.float32)
                anchor_root[new]=previous['root'][old];anchor_q[new]=previous['q'][old];active[new]=1.
                raw=human[indices[0]-1:indices[-1]+1]
                if raw.shape!=(8,22,3):raise ValueError('bad source window')
                left,right=foot_detect(raw[None],thres=(.05/120.)**2);target=jnp.asarray(raw[1:])
                root_value,joints=solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,
                    is_left_foot_contact=jnp.asarray(left[0]>0),is_right_foot_contact=jnp.asarray(right[0]>0),
                    left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),
                    smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,
                    weights=baseline['weights'],pick_point=jnp.zeros(3),anchor_roots=jnp.asarray(anchor_root),
                    anchor_joints=jnp.asarray(anchor_q),anchor_mask=jnp.asarray(active),anchor_weight=10.)
                root=np.asarray(root_value.wxyz_xyz);q=np.asarray(joints)
            if root.shape!=(7,7) or q.shape!=(7,27) or not np.isfinite(root).all() or not np.isfinite(q).all():
                raise ValueError('invalid result')
            if not np.allclose(np.linalg.norm(root[:,:4],axis=1),1.,atol=1e-4,rtol=0):raise ValueError('bad quaternion')
            mismatch=None
            if previous is not None:
                common=np.intersect1d(previous['indices'],indices);old=np.searchsorted(previous['indices'],common);new=np.searchsorted(indices,common)
                a=Rotation.from_quat(previous['root'][old][:,[1,2,3,0]]);b=Rotation.from_quat(root[new][:,[1,2,3,0]])
                mismatch={'sample_indices':common.tolist(),
                          'root_m':float(np.linalg.norm(previous['root'][old,4:]-root[new,4:],axis=1).max()),
                          'rotation_rad':float((a.inv()*b).magnitude().max()),
                          'joint_rad':float(np.abs(previous['q'][old]-q[new]).max())}
            output=HERE/f'window_{wi:02d}.npz'
            with output.open('xb') as f:
                np.savez_compressed(f,sample_indices=indices,source_frames=frames[indices],source_times=times[indices],
                                    root_wxyz_xyz=root,joint_pos=q,joint_names=np.asarray(names))
            record={'window':wi,'artifact':output.name,'artifact_sha256':sha(output),
                    'sample_indices':indices.tolist(),'overlap':mismatch}
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            previous={'indices':indices,'root':root,'q':q};records.append(record)
            print('WINDOW',wi+1,'of',len(rows),flush=True)
    if sha(SOURCE)!=baseline['source_sha256']:raise ValueError('source changed')
    out={'source_sha256':sha(SOURCE),'helper_sha256':sha(helper),'urdf_sha256':sha(URDF),
         'script_sha256':sha(Path(__file__)),'rows_sha256':sha(HERE/'windows.jsonl'),
         'source_indices_covered':[1,354], 'window_count':len(records), 'weights':baseline['weights'],
         'anchor_weight':10.,'elapsed_s':time.monotonic()-began,'training_ready':False,
         'limitations':['seed from unanchored baseline; soft weight 10 is exploratory',
                        'old v1 optimization and assumed floor; not author v2',
                        'source identity, physical quality and caption unaccepted']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
