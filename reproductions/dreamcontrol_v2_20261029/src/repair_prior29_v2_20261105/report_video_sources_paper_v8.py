"""Create four original-viewer reference playbacks and four observed PhysX replays."""
import ast
from collections import Counter
import hashlib
import json
import os
import pickle
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import jaxlie
import numpy as np
import yourdfpy

from job_guard_v1 import write

HERE = Path(__file__).resolve()
STAGE = HERE.parent.parent
ROOT = STAGE.parent
OUT = STAGE / 'report_video_sources_paper_v8'
GEN = STAGE / 'new_references_paper_v8'
SIM = STAGE / 'supermimic_paper_v8'
OLD = ROOT / 'reference_videos_upstream_20261105/v3'
URDF = OLD / 'g1_29body_14hand_viewer.urdf'
VIEWER = OLD / 'upstream_visualize_trajectories.py'
TASKS = ('pick', 'pour', 'bimanual_box_lift', 'box_push_pull')
FLOOR = 8 * 1024**3 + 356 * 1024**2


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Content changed: {path}')
    return path


def commands():
    source = ROOT / 'stage3_pick29_20261103/src/stage3_pick29_env.py'
    check(source, '5d71619407c16026e3b1f14c1862fa08ffeac2e2ffd58f2560026c3421652069')
    result = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('left_hand', 'right_hand'):
                    result[target.id] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords
                                         if kw.arg in ('open_command_expr', 'close_command_expr')}
    return result


def load_reference(task, joint_names):
    receipt_path = GEN / 'generation_receipt.json'
    receipt = read(receipt_path)
    manifest = check(GEN / 'generation_manifest.json', receipt['generation_manifest_sha256'])
    rows = [r for r in read(manifest)['records'] if r['selected_reference']]
    assert receipt['status'] == 'ready_for_eval' and Counter(r['task'] for r in rows) == dict.fromkeys(TASKS, 500)
    row = min((r for r in rows if r['task'] == task), key=lambda r: r['candidate_id'])
    assert row['admitted']
    path = check(Path(row['source']), row['source_sha256'])
    with np.load(path, allow_pickle=False) as data:
        hands = data['hand_closed']
        joints = np.zeros((len(hands), 43), dtype=np.float32)
        joints[:, [joint_names.index(name) for name in data['joint_names'].tolist()]] = data['q']
        hand_commands = commands()
        for side, col in [('left', 0), ('right', 1)]:
            for idx, name in enumerate(joint_names):
                if name.startswith(side + '_hand_'):
                    for closed, field in [(0, 'open_command_expr'), (1, 'close_command_expr')]:
                        value, = [v for pattern, v in hand_commands[side + '_hand'][field].items()
                                  if re.fullmatch(pattern, name)]
                        joints[hands[:, col] == closed, idx] = value
        pose = np.c_[data['root_pos'], data['root_quat']]
    meta = {'task': task, 'candidate_id': row['candidate_id'], 'scope': 'Generated kinematic reference and hand plan',
            'selection': 'Lowest admitted candidate ID in this task', 'source': str(path), 'source_sha256': sha(path),
            'generation_manifest_sha256': sha(manifest), 'fps': 20}
    return pose, joints, None, meta


def load_physics(task, joint_names):
    report_path = SIM / ('object_eval_' + task) / 'report.json'
    report = read(report_path)
    assert report['status'] == 'frozen_policy_task_evaluation_completed' and report['episode_count'] == 100
    trace = report['video_trace']
    path = check(Path(trace['path']), trace['sha256'])
    with np.load(path, allow_pickle=False) as data:
        names = data['joint_names'].tolist()
        assert set(names) == set(joint_names) and len(names) == len(joint_names) == 43
        joints = data['joint_position'][:, [names.index(name) for name in joint_names]]
        pose, object_pose = data['root_pose_wxyz'], data['object_pose_wxyz']
        assert len(pose) == trace['frames'] == report['episodes'][0]['steps'] + 1
        assert int(data['fps']) == trace['fps'] == 50
    assert trace['candidate_id'] == report['episodes'][0]['candidate_id']
    meta = {**trace, 'task': task, 'report': str(report_path), 'report_sha256': sha(report_path),
            'checkpoint_sha256': report['checkpoint_sha256'], 'episode_success': report['episodes'][0]['success'],
            'success_count': report['success_count'], 'episode_count': report['episode_count']}
    return pose, joints, object_pose, meta


def render(kind, task):
    from viser import _messages

    assert shutil.disk_usage(ROOT).free >= FLOOR + 64 * 1024**2
    provenance = read(OLD / 'prepared.json')
    check(VIEWER, 'b3bafd6202fe0337d05316c6759218c1b1baea0c8a02ecdd68b2c7501968ccea')
    check(URDF, provenance['adapted_urdf_sha256'])
    names = list(yourdfpy.URDF.load(str(URDF), load_meshes=False).actuated_joint_names)
    assert names == provenance['joint_names'] and len(names) == len(set(names)) == 43
    loader = load_reference if kind == 'reference' else load_physics
    pose, joints, object_pose, meta = loader(task, names)
    assert np.isfinite(pose).all() and np.isfinite(joints).all() and 2 <= len(pose) <= 1000
    dest = OUT / f'{kind}_{task}'
    dest.mkdir()
    root = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(pose[:, 3:7]),
                                                      pose[:, :3] - np.array([0., 0., .035]))
    with (dest / 'clip.pkl').open('xb') as f:
        pickle.dump({'global_pose': root, 'joints': joints}, f)
    recorder = None
    obj = None

    def begin(server):
        nonlocal recorder, obj
        if object_pose is not None:
            server.scene.add_box('/table', dimensions=tuple(meta['table_size']),
                                 position=tuple(meta['table_center']), color=(115, 127, 144))
            obj = server.scene.add_box('/observed_object', dimensions=tuple(meta['object_size']),
                                       position=tuple(object_pose[0, :3]), wxyz=tuple(object_pose[0, 3:]),
                                       color=(238, 111, 38))
        recorder = server.get_scene_serializer()
        for message in (
            _messages.SetCameraPositionMessage(position=(2.15, -2.3, 1.55)),
            _messages.SetCameraLookAtMessage(look_at=(.10, 0., .75)),
            _messages.SetCameraUpDirectionMessage(position=(0., 0., 1.)),
            _messages.SetCameraFovMessage(fov=.75),
        ):
            recorder._insert_message(message)

    def capture(server, frame):
        if obj is not None:
            obj.position = tuple(object_pose[frame, :3])
            obj.wxyz = tuple(object_pose[frame, 3:])
        recorder.insert_sleep(1 / meta['fps'])

    def finish(server):
        target = dest / 'clip.viser'
        with target.open('xb') as f:
            f.write(recorder.serialize())
        write(dest / 'receipt.json', {**meta, 'kind': kind, 'frames': len(pose),
              'duration_seconds': len(pose) / meta['fps'], 'viser_path': str(target),
              'viser_sha256': sha(target), 'viser_bytes': target.stat().st_size,
              'viewer_sha256': sha(VIEWER), 'urdf_sha256': sha(URDF)})
        server.stop()

    source = VIEWER.read_text()

    def change(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    change("yourdfpy.URDF.load('../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')",
           f'yourdfpy.URDF.load({str(URDF)!r})')
    change('server = viser.ViserServer()', "server = viser.ViserServer(host='127.0.0.1', port=18091)")
    change('while True:', 'begin(server)\n    for export_frame in range(num_timesteps):')
    change('timestep_slider.value = (timestep_slider.value + 1) % num_timesteps',
           'timestep_slider.value = export_frame')
    change('time.sleep(1./20.)', 'capture(server, tstep)\n    finish(server)')
    change('onp.zeros((1000, 1000)', 'onp.zeros((100, 100)')
    change('jnp.array([0.01, 0.01, 1.0])', 'jnp.array([0.1, 0.1, 1.0])')
    (dest / 'viewer_executed.py').write_text(source)
    sys.argv = [str(VIEWER), '--folder', str(dest), '--id', 'clip']
    exec(compile(source, str(dest / 'viewer_executed.py'), 'exec'),
         {'__name__': '__main__', 'begin': begin, 'capture': capture, 'finish': finish})


def controller():
    OUT.mkdir()
    status = OUT / 'status.json'
    completed = {}
    try:
        while len(completed) < 8:
            for kind in ('reference', 'physics'):
                for task in TASKS:
                    key = f'{kind}_{task}'
                    if key in completed:
                        continue
                    ready = GEN / 'generation_receipt.json' if kind == 'reference' else SIM / ('object_eval_' + task) / 'report.json'
                    if not ready.is_file():
                        continue
                    if shutil.disk_usage(ROOT).free < FLOOR + 64 * 1024**2:
                        continue
                    write(status, {'status': 'serializing', 'active': key, 'completed': completed})
                    with (OUT / f'{key}.log').open('xb') as log:
                        subprocess.run([sys.executable, '-B', str(HERE), kind, task], check=True,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=300)
                    receipt = OUT / key / 'receipt.json'
                    completed[key] = {'receipt': str(receipt), 'sha256': sha(receipt)}
            for state in (GEN / 'status.json', SIM / 'status.json'):
                if state.exists() and read(state)['status'] == 'failed_preserved':
                    raise RuntimeError(f'Upstream failed: {state}')
            write(status, {'status': 'waiting_for_reports' if len(completed) < 8 else 'eight_sources_ready',
                           'completed': completed})
            if len(completed) < 8:
                time.sleep(30)
    except BaseException:
        write(status, {'status': 'failed_preserved', 'completed': completed, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    if sys.argv[1:]:
        render(*sys.argv[1:])
    else:
        controller()

