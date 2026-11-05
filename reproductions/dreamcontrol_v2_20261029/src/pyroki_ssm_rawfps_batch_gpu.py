"""Retarget three source-rate SSM clips with the isolated approximate G1 solver.

Window solves, temporal weights, assumed plane and the ankle-speed proxy are
exploratory adaptations of earlier code. Outputs require subsequent audit and
are never themselves accepted as training data.
"""
import argparse
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
VARIANT = ROOT / 'pyroki_grab_temporal_20261030'
HUMAN = ROOT / 'ssm_rawfps_human_batch_20261031'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof, create_conn_tree, solve_retargeting, foot_detect


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def window_starts(count):
    if not isinstance(count, int) or count < 8:
        raise ValueError('at least eight source samples required')
    starts = list(range(0, count - 7, 4))
    if starts[-1] != count - 8:
        starts.append(count - 8)
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-root', type=Path, default=HUMAN)
    parser.add_argument('--samples', nargs='+', default=['007410','011120','002941'])
    args = parser.parse_args()
    if len(args.samples) != len(set(args.samples)) or any(not s.isdecimal() or len(s)!=6 for s in args.samples):
        raise ValueError('unique six-digit sample IDs required')
    human_root = args.human_root.resolve(strict=True)
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise ValueError('one explicitly selected GPU required')
    helper = VARIANT/'retarget_helpers/_utils.py'; receipt = json.loads((VARIANT/'report.json').read_text())
    urdf_path = PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if sha(helper) != receipt['helper_sha256'] or sha(urdf_path) != receipt['urdf_sha256']:
        raise ValueError('solver helper or URDF changed')
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, load_collision_meshes=True)
    robot = pk.Robot.from_urdf(urdf); collision = pk.collision.RobotCollision.from_urdf(urdf)
    hi, ri = get_humanoid_retarget_indices_g1_27dof(); mask = create_conn_tree(robot, ri)
    plane = pk.collision.Heightmap(pose=jaxlie.SE3.identity(), size=jnp.array([.01,.01,1.]),
                                   height_data=jnp.zeros((100,100), dtype=jnp.float32))
    names = list(robot.joints.actuated_names)
    progress = HERE/'progress.jsonl'
    with progress.open('x', encoding='utf8', buffering=1) as stream:
        for sample in args.samples:
            folder = human_root/sample; human_report = json.loads((folder/'human_report.json').read_text())
            checked = json.loads((folder/'human_audit.json').read_text()); source = folder/'human_rawfps.npz'
            if (sha(folder/'human_report.json') != checked['human_report_sha256'] or
                sha(source) != checked['artifact_sha256'] or sha(source) != human_report['artifact_sha256']):
                raise ValueError('audited human input changed')
            with np.load(source,allow_pickle=False) as f:
                human = f['human'].astype(np.float32); source_frames = f['source_frames'].copy(); times = f['source_times'].copy()
            fps = human_report['reference_fps']
            if (human.shape != (len(times),22,3) or not np.isfinite(human).all() or
                not np.array_equal(np.diff(source_frames), np.ones(len(source_frames)-1, dtype=int)) or
                not np.allclose(times, source_frames/fps, rtol=0, atol=1e-12) or abs(fps-60.) > .001):
                raise ValueError('unsupported input timeline')
            weights = dict(receipt['weights'])
            weights['root_smoothness'] = weights['foot_skating'] = 2.0 * fps / 20.0
            starts = window_starts(len(human))
            out = HERE/sample; out.mkdir(exist_ok=False)
            rows, previous = [], None; began = time.monotonic()
            with (out/'windows.jsonl').open('x', encoding='utf8', buffering=1) as log:
                for wi, start in enumerate(starts):
                    w = human[start:start+8]; indices = np.arange(start+1,start+8)
                    left, right = foot_detect(w[None], thres=(.05/fps)**2)
                    target = jnp.asarray(w[1:])
                    root, joints = solve_retargeting(robot=robot, robot_coll=collision, target_keypoints=target,
                        is_left_foot_contact=jnp.asarray(left[0]>0), is_right_foot_contact=jnp.asarray(right[0]>0),
                        left_foot_keypoints=plane.project_points(target[:,10,:]),
                        right_foot_keypoints=plane.project_points(target[:,11,:]),
                        smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,
                        heightmap=plane,weights=weights,pick_point=jnp.zeros(3))
                    q, rq = np.asarray(joints), np.asarray(root.wxyz_xyz)
                    if (q.shape != (7,27) or rq.shape != (7,7) or not np.isfinite(q).all() or not np.isfinite(rq).all() or
                        not np.allclose(np.linalg.norm(rq[:,:4],axis=1),1.,atol=1e-4,rtol=0)):
                        raise ValueError('invalid solver output')
                    overlap = None
                    if previous is not None:
                        common = np.intersect1d(previous['indices'],indices)
                        old = np.searchsorted(previous['indices'],common); new = np.searchsorted(indices,common)
                        a = Rotation.from_quat(previous['root'][old][:,[1,2,3,0]])
                        b = Rotation.from_quat(rq[new][:,[1,2,3,0]])
                        overlap = {'sample_indices': common.tolist(),
                                   'root_m': float(np.linalg.norm(previous['root'][old,4:]-rq[new,4:],axis=1).max()),
                                   'rotation_rad': float((a.inv()*b).magnitude().max()),
                                   'joint_rad': float(np.abs(previous['q'][old]-q[new]).max())}
                    output = out / f'window_{wi:03d}.npz'
                    with output.open('xb') as f:
                        np.savez_compressed(f,sample_indices=indices,source_frames=source_frames[indices],
                            source_times=times[indices],root_wxyz_xyz=rq,joint_pos=q,joint_names=np.asarray(names))
                    row = {'window':wi,'artifact':output.name,'artifact_sha256':sha(output),
                           'sample_indices':indices.tolist(),'overlap':overlap}
                    log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush(); os.fsync(log.fileno())
                    previous = {'indices':indices,'root':rq,'q':q}; rows.append(row)
                    print('WINDOW',sample,wi+1,'of',len(starts),flush=True)
            if sha(source) != checked['artifact_sha256']:
                raise RuntimeError('source changed during solve')
            report = {'id':sample,'backend':'gpu','source_sha256':sha(source),
                      'human_report_sha256':sha(folder/'human_report.json'),'human_audit_sha256':sha(folder/'human_audit.json'),
                      'helper_sha256':sha(helper),'urdf_sha256':sha(urdf_path),'script_sha256':sha(Path(__file__)),
                      'reference_fps':fps,'weights':weights,'source_sample_indices':[1,len(human)-1],
                      'source_raw_frame_first_last':[int(source_frames[1]),int(source_frames[-1])],
                      'window_count':len(rows),'rows_sha256':sha(out/'windows.jsonl'),
                      'elapsed_s':time.monotonic()-began,'training_ready':False,
                      'limitations':['Old v1 approximation, heuristic source-rate temporal scaling, assumed z=0.',
                      'Window solutions are not stitched and are not accepted for training.',
                      'Source identity, true FPS, captions and physical quality remain unaccepted.']}
            with (out/'report.json').open('x',encoding='utf8') as f:
                json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
            stream.write(json.dumps({'id':sample,'window_count':len(rows),'report_sha256':sha(out/'report.json')})+'\n')
            stream.flush(); os.fsync(stream.fileno()); print('COMPLETE',sample,flush=True)


if __name__ == '__main__': main()
