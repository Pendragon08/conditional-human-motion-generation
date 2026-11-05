"""Compare soft-overlap PyRoki continuation on audited raw-rate SSM clips.

This is an exploratory warm continuity constraint, not the v2 authors' loss.
Windows, captions, true ground and physics remain unaccepted for training.
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

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
HUMAN = ROOT/'ssm_rawfps_human_batch_20261031'
BASE = ROOT/'ssm_rawfps_gpu_batch_20261031'
ANCHOR = ROOT/'pyroki_s3_anchor_tail_20261030'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof, create_conn_tree, solve_retargeting, foot_detect


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def build_overlap_anchor(previous, indices):
    common = np.intersect1d(previous['indices'],indices)
    old = np.searchsorted(previous['indices'],common); new = np.searchsorted(indices,common)
    if len(common) not in (3,4,5,6) or not np.array_equal(new,np.arange(len(common))):
        raise ValueError('invalid overlap')
    ar = np.zeros((7,7),dtype=np.float32); ar[:,0] = 1.
    aq = np.zeros((7,27),dtype=np.float32); active = np.zeros(7,dtype=np.float32)
    ar[new] = previous['root'][old]; aq[new] = previous['q'][old]; active[new] = 1.
    return ar, aq, active


def main():
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise ValueError('one explicitly selected GPU required')
    anchor_manifest = json.loads((ANCHOR/'manifest.json').read_text())
    helper = HERE/'retarget_helpers/_utils.py'
    baseline_helper = ROOT/'pyroki_grab_temporal_20261030/retarget_helpers/_utils.py'
    base_receipt = json.loads((ROOT/'pyroki_grab_temporal_20261030/report.json').read_text())
    urdf = PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if (sha(helper) != anchor_manifest['anchor_helper_sha256'] or
        sha(baseline_helper) != base_receipt['helper_sha256'] or sha(urdf) != base_receipt['urdf_sha256']):
        raise ValueError('solver helper or URDF changed')
    model = yourdfpy.URDF.load(str(urdf),load_meshes=False,load_collision_meshes=True)
    robot = pk.Robot.from_urdf(model); collision = pk.collision.RobotCollision.from_urdf(model)
    hi, ri = get_humanoid_retarget_indices_g1_27dof(); mask = create_conn_tree(robot, ri)
    plane = pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),
                                   height_data=jnp.zeros((100,100),dtype=jnp.float32))
    names = list(robot.joints.actuated_names)
    with (HERE/'progress.jsonl').open('x',encoding='utf8',buffering=1) as progress:
        for sample in ('007410','011120','002941'):
            source_dir = HUMAN/sample; base = BASE/sample
            source = source_dir/'human_rawfps.npz'
            hr = json.loads((source_dir/'human_report.json').read_text())
            human_audit = json.loads((source_dir/'human_audit.json').read_text())
            baseline = json.loads((base/'report.json').read_text()); check = json.loads((base/'audit_report.json').read_text())
            if (sha(source) != human_audit['artifact_sha256'] or sha(source) != baseline['source_sha256'] or
                sha(base/'report.json') != check['report_sha256'] or sha(base/'windows.jsonl') != baseline['rows_sha256'] or
                sha(urdf) != baseline['urdf_sha256'] or sha(baseline_helper) != baseline['helper_sha256']):
                raise ValueError('baseline provenance mismatch')
            rows = [json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
            if len(rows) != baseline['window_count'] or len(rows) != check['windows_verified']:
                raise ValueError('baseline incomplete')
            with np.load(source,allow_pickle=False) as f:
                human = f['human'].astype(np.float32); frames = f['source_frames'].copy(); times = f['source_times'].copy()
            fps = float(hr['reference_fps'])
            if (human.shape != (len(frames),22,3) or not np.isfinite(human).all() or abs(fps-60.)>.001 or
                not np.array_equal(np.diff(frames),np.ones(len(frames)-1,dtype=int)) or
                not np.allclose(times,frames/fps,atol=1e-12,rtol=0)):
                raise ValueError('invalid source timeline')
            output_dir = HERE/sample; output_dir.mkdir(exist_ok=False)
            written, previous = [], None; began = time.monotonic()
            with (output_dir/'windows.jsonl').open('x',encoding='utf8',buffering=1) as log:
                for wi, row in enumerate(rows):
                    indices = np.asarray(row['sample_indices'],dtype=int)
                    if row['window'] != wi or len(indices) != 7 or not np.array_equal(indices,np.arange(indices[0],indices[0]+7)):
                        raise ValueError('baseline window indexing mismatch')
                    if wi == 0:
                        seed = base/row['artifact']
                        if sha(seed) != row['artifact_sha256']:
                            raise ValueError('first-window seed changed')
                        with np.load(seed,allow_pickle=False) as f:
                            root = f['root_wxyz_xyz'].copy(); q = f['joint_pos'].copy()
                            if [str(x) for x in f['joint_names']] != names:
                                raise ValueError('joint names changed')
                    else:
                        ar, aq, active = build_overlap_anchor(previous,indices)
                        window = human[indices[0]-1:indices[-1]+1]
                        if window.shape != (8,22,3): raise ValueError('bad source window')
                        left,right = foot_detect(window[None],thres=(.05/fps)**2); target = jnp.asarray(window[1:])
                        rootvalue,joints = solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,
                            is_left_foot_contact=jnp.asarray(left[0]>0),is_right_foot_contact=jnp.asarray(right[0]>0),
                            left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),
                            smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,
                            weights=baseline['weights'],pick_point=jnp.zeros(3),anchor_roots=jnp.asarray(ar),
                            anchor_joints=jnp.asarray(aq),anchor_mask=jnp.asarray(active),anchor_weight=10.)
                        root = np.asarray(rootvalue.wxyz_xyz); q = np.asarray(joints)
                    if (root.shape != (7,7) or q.shape != (7,27) or not np.isfinite(root).all() or not np.isfinite(q).all() or
                        not np.allclose(np.linalg.norm(root[:,:4],axis=1),1.,atol=1e-4,rtol=0)):
                        raise ValueError('invalid result')
                    overlap = None
                    if previous is not None:
                        common = np.intersect1d(previous['indices'],indices)
                        old = np.searchsorted(previous['indices'],common); new = np.searchsorted(indices,common)
                        a = Rotation.from_quat(previous['root'][old][:,[1,2,3,0]])
                        b = Rotation.from_quat(root[new][:,[1,2,3,0]])
                        overlap = {'sample_indices':common.tolist(),
                            'root_m':float(np.linalg.norm(previous['root'][old,4:]-root[new,4:],axis=1).max()),
                            'rotation_rad':float((a.inv()*b).magnitude().max()),
                            'joint_rad':float(np.abs(previous['q'][old]-q[new]).max())}
                    artifact = output_dir/f'window_{wi:03d}.npz'
                    with artifact.open('xb') as f:
                        np.savez_compressed(f,sample_indices=indices,source_frames=frames[indices],source_times=times[indices],
                                            root_wxyz_xyz=root,joint_pos=q,joint_names=np.asarray(names))
                    item = {'window':wi,'artifact':artifact.name,'artifact_sha256':sha(artifact),
                            'sample_indices':indices.tolist(),'overlap':overlap}
                    log.write(json.dumps(item,allow_nan=False)+'\n'); log.flush(); os.fsync(log.fileno())
                    previous = {'indices':indices,'root':root,'q':q}; written.append(item)
                    print('WINDOW',sample,wi+1,'of',len(rows),flush=True)
            if sha(source) != baseline['source_sha256']:
                raise RuntimeError('source changed during solve')
            report = {'id':sample,'backend':'gpu','source_sha256':sha(source),'human_report_sha256':sha(source_dir/'human_report.json'),
                      'baseline_report_sha256':sha(base/'report.json'),'baseline_audit_sha256':sha(base/'audit_report.json'),
                      'helper_sha256':sha(helper),'urdf_sha256':sha(urdf),'script_sha256':sha(Path(__file__)),
                      'reference_fps':fps,'weights':baseline['weights'],'anchor_weight':10.,
                      'source_sample_indices':[1,len(human)-1], 'window_count':len(written),
                      'rows_sha256':sha(output_dir/'windows.jsonl'),'elapsed_s':time.monotonic()-began,
                      'training_ready':False,'limitations':['First window copied from baseline; soft overlap weight is exploratory.',
                      'Old v1 approximation and assumed ground are not author v2.',
                      'No source time truth, caption, contact or physical acceptance.']}
            with (output_dir/'report.json').open('x',encoding='utf8') as f:
                json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
            progress.write(json.dumps({'id':sample,'window_count':len(written),'report_sha256':sha(output_dir/'report.json')})+'\n')
            progress.flush(); os.fsync(progress.fileno()); print('COMPLETE',sample,flush=True)


if __name__ == '__main__': main()
