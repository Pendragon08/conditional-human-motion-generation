"""Private replay of all observed dataset-screen episodes, including failures.

Uses the same rigid Kabsch reconstruction as render_phygile_records.py. No
reference is posed as actual execution, no physics steps or missing frames added.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    summary = json.loads((a.records/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'DIAGNOSTIC_SCREEN_COMPLETED':
        raise ValueError('Expected completed diagnostic, including every failed episode')
    contract_path = a.records/'prepared_private/contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True, exist_ok=False)
    spec = mujoco.MjSpec.from_file(str(a.model))
    spec.worldbody.add_light(name='screen_video_light', pos=[0, -3, 5], dir=[0, .3, -1], diffuse=[.9, .9, .9])
    model = spec.compile(); model.vis.global_.offwidth = 800; model.vis.global_.offheight = 560
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) != 1:
        raise ValueError('Visual asset must provide one existing floor plane')
    model.geom_matid[planes] = -1
    model.geom_rgba[planes] = [.24, .28, .32, 1]
    model.vis.quality.shadowsize = 1024
    data = mujoco.MjData(model)
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in contract['tracked_body_names']]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in contract['joint_names']]
    if min(bodies+joints) < 0:
        raise ValueError('Visual asset misses recorded bodies/joints')
    addresses = model.jnt_qposadr[joints]
    renderer = mujoco.Renderer(model, height=560, width=800)
    camera = mujoco.MjvCamera(); camera.azimuth = 135; camera.elevation = -15; camera.distance = 3.1
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    output = a.output/'PhyGile_SSM_actual_diagnostic_PRIVATE.mp4'
    writer = imageio.get_writer(str(output), format='FFMPEG', fps=50, codec='libx264', quality=8, macro_block_size=1)
    receipt = {'mode': 'private visual replay of actual measured states; zero physics steps',
        'source_summary_sha256': sha(a.records/'summary.json'), 'renderer_source_sha256': sha(__file__),
        'model_sha256': sha(a.model), 'contract_sha256': sha(contract_path),
        'root_rotation': 'rigid Kabsch reconstruction from measured body positions and joint angles',
        'fk_reconstruction_tolerance_m': .002, 'paper_benchmark': False,
        'visual_only': {'one_existing_floor': True, 'floor_rgba': [.24, .28, .32, 1], 'shadows': False},
        'trained_generation': False, 'public_redistribution': False, 'episodes': []}
    count = 0
    try:
        for row in summary['records']:
            path = a.records/f"episode_{row['index']:03d}/raw_pre_reset.npz"
            if sha(path) != row['raw_sha256']:
                raise ValueError('Measured trajectory changed')
            errors = []
            with np.load(path, allow_pickle=False) as raw:
                length = len(raw['frame'])
                for frame, (q, target) in enumerate(zip(raw['robot_joint_pos'], raw['robot_body_pos'])):
                    data.qpos[:] = model.qpos0; data.qpos[:3] = 0; data.qpos[3:7] = [1, 0, 0, 0]
                    data.qpos[addresses] = q; mujoco.mj_forward(model, data)
                    source = data.xpos[bodies].copy()
                    ac = source-source.mean(0); bc = target-target.mean(0)
                    u, _, vt = np.linalg.svd(ac.T@bc)
                    parity = np.eye(3); parity[2, 2] = np.linalg.det(u@vt)
                    rotation = u@parity@vt
                    translation = target.mean(0)-source.mean(0)@rotation
                    quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.T.flatten())
                    data.qpos[:3] = translation; data.qpos[3:7] = quat; mujoco.mj_forward(model, data)
                    error = float(np.max(np.linalg.norm(data.xpos[bodies]-target, axis=-1)))
                    errors.append(error)
                    if error > .002:
                        raise ValueError(f'Pose reconstruction exceeded tolerance: {error}')
                    camera.lookat[:] = [data.qpos[0], data.qpos[1], .8]
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    canvas = Image.new('RGB', (800, 680), (15, 22, 33))
                    canvas.paste(Image.fromarray(renderer.render()), (0, 80))
                    draw = ImageDraw.Draw(canvas)
                    label = 'FULL CLIP' if row['receipt']['full_clip_success'] else 'EARLY TERMINATION'
                    draw.text((15, 8), f"SSM pair {row['index']+1}/{len(summary['records'])}: {row['motion_id']} | {row['split']}", font=font, fill='white')
                    draw.text((15, 35), 'Frozen pilot policy: 2,500 PPO iterations | decoded 262D reference', font=small, fill=(100, 210, 255))
                    draw.text((15, 58), f"{label} | observed {length}/{row['clock']['control_frames']} control frames", font=small, fill=(255, 155, 125))
                    draw.text((15, 648), f'Actual frame {frame+1}/{length}, t={(frame+1)/50:.2f}s | private diagnostic; not paper benchmark', font=small, fill='white')
                    if count == 0:
                        canvas.save(a.output/'private_preview.png')
                    writer.append_data(np.asarray(canvas)); count += 1
            receipt['episodes'].append({'index': row['index'], 'motion_id': row['motion_id'],
                'raw_sha256': sha(path), 'start_video_frame': count-length, 'frames': length,
                'full_clip_success': row['receipt']['full_clip_success'], 'fk_reconstruction_max_m': max(errors)})
    finally:
        writer.close(); renderer.close()
    reader = imageio.get_reader(str(output))
    decoded_frames = sum(1 for _ in reader); reader.close()
    if decoded_frames != count or count != summary['observed_control_frames']:
        raise ValueError('Encoded video frame count differs from actual observations')
    receipt.update(output=str(output), video_sha256=sha(output), output_frames=count,
                   decoded_video_frames=decoded_frames, fps=50, all_episodes_included=True)
    (a.output/'render_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'episodes'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['records', 'model', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
