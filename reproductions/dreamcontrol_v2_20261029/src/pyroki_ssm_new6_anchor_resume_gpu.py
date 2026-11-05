"""Resume a verified prefix of source-rate soft-overlap G1 solver windows.

Window solves, temporal weights, assumed plane and the ankle-speed proxy are
exploratory adaptations of earlier code. Outputs require subsequent audit and
are never themselves accepted as training data.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
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
ANCHOR = ROOT / 'pyroki_s3_anchor_tail_20261030'
HUMAN = ROOT / 'ssm_rawfps_human_batch_20261031'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ANCHOR))
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


def anchor_for(previous, indices):
    root=np.zeros((7,7),dtype=np.float32);root[:,0]=1.
    joints=np.zeros((7,27),dtype=np.float32);active=np.zeros(7,dtype=np.float32)
    if previous is not None:
        common=np.intersect1d(previous['indices'],indices)
        old=np.searchsorted(previous['indices'],common);new=np.searchsorted(indices,common)
        if len(common) not in (3,4,5,6) or not np.array_equal(new,np.arange(len(common))):
            raise ValueError('invalid window overlap')
        root[new]=previous['root'][old];joints[new]=previous['q'][old];active[new]=1.
    return root,joints,active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-root', type=Path, default=HUMAN)
    parser.add_argument('--samples', nargs='+', default=['007410','011120','002941'])
    args = parser.parse_args()
    if len(args.samples) != len(set(args.samples)) or any(not s.isdecimal() or len(s)!=6 for s in args.samples):
        raise ValueError('unique six-digit sample IDs required')
    human_root = args.human_root.resolve(strict=True)
    precheck_path=HERE/'resume_precheck.json';precheck=json.loads(precheck_path.read_text())
    prior=Path(precheck['old_run']).resolve(strict=True)
    if (sha(prior/'pyroki_ssm_new6_anchor_gpu.py')!=precheck['old_script_sha256'] or
        sha(prior/'queue_status.txt')!=precheck['old_status_sha256'] or
        sha(prior/'progress.jsonl')!=precheck['old_progress_sha256'] or
        sha(HERE/'progress.jsonl')!=precheck['old_progress_sha256']):
        raise ValueError('prior prefix changed after precheck')
    prefix={x['id']:x for x in precheck['cases']}
    if set(prefix)!=set(args.samples):raise ValueError('precheck coverage differs from requested samples')
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise ValueError('one explicitly selected GPU required')
    helper = ANCHOR/'retarget_helpers/_utils.py'; receipt = json.loads((VARIANT/'report.json').read_text())
    anchor_manifest=json.loads((ANCHOR/'manifest.json').read_text())
    baseline_helper=VARIANT/'retarget_helpers/_utils.py'
    urdf_path = PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if (sha(helper) != anchor_manifest['anchor_helper_sha256'] or
        sha(baseline_helper) != receipt['helper_sha256'] or sha(urdf_path) != receipt['urdf_sha256']):
        raise ValueError('solver helper or URDF changed')
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, load_collision_meshes=True)
    robot = pk.Robot.from_urdf(urdf); collision = pk.collision.RobotCollision.from_urdf(urdf)
    hi, ri = get_humanoid_retarget_indices_g1_27dof(); mask = create_conn_tree(robot, ri)
    plane = pk.collision.Heightmap(pose=jaxlie.SE3.identity(), size=jnp.array([.01,.01,1.]),
                                   height_data=jnp.zeros((100,100), dtype=jnp.float32))
    names = list(robot.joints.actuated_names)
    progress = HERE/'progress.jsonl'
    with progress.open('a', encoding='utf8', buffering=1) as stream:
        for sample in args.samples:
            folder = human_root/sample; human_report = json.loads((folder/'human_report.json').read_text())
            checked = json.loads((folder/'human_audit.json').read_text()); source = folder/'human_rawfps.npz'
            if (sha(folder/'human_report.json') != checked['human_report_sha256'] or
                sha(source) != checked['artifact_sha256'] or sha(source) != human_report['artifact_sha256'] or
                sha(source) != prefix[sample]['source_sha256']):
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
            out = HERE/sample; initial=prefix[sample]['window_count']
            if prefix[sample]['status']=='complete':
                prior_progress=next(x for x in (json.loads(s) for s in (prior/'progress.jsonl').read_text().splitlines()) if x['id']==sample)
                if sha(out/'report.json')!=prior_progress['report_sha256'] or sha(out/'windows.jsonl')!=prefix[sample]['rows_sha256']:
                    raise ValueError('copied completed sample changed')
                print('REUSED',sample,initial,'windows',flush=True);continue
            rows,previous=[],None
            if prefix[sample]['status']=='partial':
                if sha(out/'windows.jsonl')!=prefix[sample]['rows_sha256']:raise ValueError('copied partial prefix changed')
                rows=[json.loads(s) for s in (out/'windows.jsonl').read_text().splitlines()]
                if len(rows)!=initial or len(rows)>=len(starts):raise ValueError('invalid partial prefix length')
                last=out/rows[-1]['artifact']
                if sha(last)!=rows[-1]['artifact_sha256']:raise ValueError('last window changed')
                with np.load(last,allow_pickle=False) as f:
                    previous={'indices':f['sample_indices'].copy(),'root':f['root_wxyz_xyz'].copy(),'q':f['joint_pos'].copy()}
                mode='a'
            elif prefix[sample]['status']=='not_started':
                out.mkdir(exist_ok=False);mode='x'
            else:raise ValueError('unknown prefix status')
            began = time.monotonic()
            with (out/'windows.jsonl').open(mode, encoding='utf8', buffering=1) as log:
                for wi in range(len(rows),len(starts)):
                    start=starts[wi]
                    if wi and wi%10==0:
                        # The prior JAX result is materialized as NumPy below;
                        # wait for the utilization sampling period before
                        # checking whether other GPU work has become busy.
                        time.sleep(1.2)
                        values=subprocess.check_output(['nvidia-smi','-i','1',
                            '--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
                        utilization,free_mb=map(int,values)
                        if utilization>30 or free_mb<2048 or shutil.disk_usage(ROOT).free<8*1024**3:
                            raise RuntimeError(f'resource guard stopped: utilization={utilization}%, GPU free={free_mb}MiB')
                    w = human[start:start+8]; indices = np.arange(start+1,start+8)
                    left, right = foot_detect(w[None], thres=(.05/fps)**2)
                    target = jnp.asarray(w[1:])
                    anchor_roots,anchor_joints,anchor_mask=anchor_for(previous,indices)
                    root, joints = solve_retargeting(robot=robot, robot_coll=collision, target_keypoints=target,
                        is_left_foot_contact=jnp.asarray(left[0]>0), is_right_foot_contact=jnp.asarray(right[0]>0),
                        left_foot_keypoints=plane.project_points(target[:,10,:]),
                        right_foot_keypoints=plane.project_points(target[:,11,:]),
                        smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,
                        heightmap=plane,weights=weights,pick_point=jnp.zeros(3),
                        anchor_roots=jnp.asarray(anchor_roots),anchor_joints=jnp.asarray(anchor_joints),
                        anchor_mask=jnp.asarray(anchor_mask),anchor_weight=10.)
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
                      'helper_sha256':sha(helper),'baseline_helper_sha256':sha(baseline_helper),
                      'anchor_weight':10.,'urdf_sha256':sha(urdf_path),'script_sha256':sha(Path(__file__)),
                      'resume_precheck_sha256':sha(precheck_path),'prefix_windows_reused':initial,
                      'reference_fps':fps,'weights':weights,'source_sample_indices':[1,len(human)-1],
                      'source_raw_frame_first_last':[int(source_frames[1]),int(source_frames[-1])],
                      'window_count':len(rows),'rows_sha256':sha(out/'windows.jsonl'),
                      'elapsed_s':time.monotonic()-began,'training_ready':False,
                      'limitations':['Old v1 approximation, heuristic source-rate temporal scaling, assumed z=0 and soft window anchors.',
                      'Window solutions are not stitched and are not accepted for training.',
                      'Source identity, true FPS, captions and physical quality remain unaccepted.']}
            with (out/'report.json').open('x',encoding='utf8') as f:
                json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
            stream.write(json.dumps({'id':sample,'window_count':len(rows),'report_sha256':sha(out/'report.json')})+'\n')
            stream.flush(); os.fsync(stream.fileno()); print('COMPLETE',sample,flush=True)


if __name__ == '__main__': main()
