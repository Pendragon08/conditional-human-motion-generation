"""Private actual-state video of every pair at one prespecified seed; no physics."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((a.records/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'AUXILIARY_PHYSICAL_SCREEN_COMPLETED':
        raise ValueError('A completed physical batch, including failures, is required')
    rows = [r for r in summary['records'] if r['seed'] == a.seed]
    if len(rows) != summary['selected_sequences'] or len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Every source pair must be included once at the chosen seed')
    spec = mujoco.MjSpec.from_file(str(a.model))
    spec.worldbody.add_light(name='filter_video_light', pos=[0, -3, 5], dir=[0, .3, -1], diffuse=[.9, .9, .9])
    model = spec.compile()
    model.vis.global_.offwidth = 800;model.vis.global_.offheight = 560
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) != 1:
        raise ValueError('Exactly one existing visual floor is required')
    model.geom_matid[planes] = -1;model.geom_rgba[planes] = [.24, .28, .32, 1]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=560, width=800)
    camera = mujoco.MjvCamera();camera.azimuth = 135;camera.elevation = -15;camera.distance = 3.1
    option = mujoco.MjvOption();option.geomgroup[3] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    video = a.output/'PhyGile_pairs_external_ScaleBFM_actual_ALL40_PRIVATE.mp4'
    writer = imageio.get_writer(str(video), format='FFMPEG', fps=50, codec='libx264', quality=8, macro_block_size=1)
    receipt = {'status': 'RENDERING', 'role': 'private replay of actual external-tracker screening',
        'seed': a.seed, 'trained_generator': False, 'paper_phygile_gmt': False,
        'physics_steps_added': 0, 'public_redistribution': False,
        'summary_sha256': sha(a.records/'summary.json'), 'model_sha256': sha(a.model),
        'renderer_sha256': sha(__file__), 'episodes': []}
    count = 0
    try:
        for row in rows:
            raw = a.records/f"pair_{row['index']:03d}_seed_{a.seed}/actual_pre_reset.npz"
            if sha(raw) != row['raw_sha256']:
                raise ValueError('Original actual states changed')
            residuals = []
            with np.load(raw, allow_pickle=False) as x:
                qpos, targets = x['qpos'], x['body_pos']
                names = x['body_names'].tolist()
                body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
                if min(body_ids) < 0 or qpos.shape[1] != model.nq or len(qpos) != row['control_frames']:
                    raise ValueError('Visual model/record contract mismatch')
                for frame, (q, target) in enumerate(zip(qpos, targets)):
                    data.qpos[:] = q;mujoco.mj_forward(model, data)
                    residual = float(np.linalg.norm(data.xpos[body_ids]-target, axis=-1).max())
                    residuals.append(residual)
                    if residual > .002:
                        raise ValueError('Visual FK differs from the actual recorded geometry')
                    camera.lookat[:] = [q[0], q[1], .8]
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    canvas = Image.new('RGB', (800, 680), (15, 22, 33))
                    canvas.paste(Image.fromarray(renderer.render()), (0, 80))
                    draw = ImageDraw.Draw(canvas)
                    result = ('PASS' if row['auxiliary_pass'] else 'EARLY FAILURE' if row['terminal_failure'] else 'FULL CLIP / ERROR LIMIT')
                    draw.text((15, 8), f"SSM pair {row['index']+1:02d}/{len(rows)} | {row['split']} | seed {a.seed} | {result}", font=font, fill='white')
                    draw.text((15, 35), 'Frozen external ScaleBFM | auxiliary screen, not PhyGile GMT', font=small, fill=(100, 210, 255))
                    draw.text((15, 58), f"Observed {len(qpos)}/{row['reference_frames']} frames | mean MPJPE {row['mean_mpjpe_m']*1000:.1f} mm", font=small, fill=(255, 155, 125))
                    draw.text((15, 648), f'Actual frame {frame+1}/{len(qpos)} | 50 fps | all failures retained | PRIVATE', font=small, fill='white')
                    if count == 0:
                        canvas.save(a.output/'private_preview.png')
                    writer.append_data(np.asarray(canvas));count += 1
            receipt['episodes'].append({'index': row['index'], 'seed': a.seed, 'frames': row['control_frames'],
                'start_video_frame': count-row['control_frames'], 'auxiliary_pass': row['auxiliary_pass'],
                'raw_sha256': row['raw_sha256'], 'maximum_fk_residual_m': max(residuals)})
        receipt['status'] = 'RENDERED'
    except BaseException as exc:
        receipt.update(status='FAILED', error=repr(exc))
        raise
    finally:
        writer.close();renderer.close()
        receipt['written_frames'] = count
        (a.output/'render_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    reader = imageio.get_reader(str(video))
    decoded = sum(1 for _ in reader);meta = reader.get_meta_data();reader.close()
    if decoded != count or count != sum(r['control_frames'] for r in rows) or abs(meta['fps']-50) > 1e-6:
        raise ValueError('Encoded video changed/dropped actual frame timing')
    receipt.update(status='ACTUAL_STATE_VIDEO_VERIFIED', decoded_frames=decoded, fps=50,
                   video_sha256=sha(video), video_bytes=video.stat().st_size, output=str(video))
    (a.output/'render_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'episodes'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['records', 'model', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
