"""Reconstruct marker-aligned SSM candidate at reference raw frame rate."""
import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
import zipfile

import numpy as np
import smplx
import torch

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
CATALOG = ROOT / 'ssm_new_source_shortlist_v3_20261031/report.json'
MODEL = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=True)
    p.add_argument('--catalog-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    if sha(CATALOG) != a.catalog_sha256:
        raise ValueError('candidate catalog changed')
    catalog = json.loads(CATALOG.read_text()); rows = [x for x in catalog['candidates'] if x['id'] == a.id]
    if len(rows) != 1:
        raise ValueError('candidate ID absent or not unique')
    c = rows[0]; source = Path(c['source']); ref = Path(c['reference']); receipt_path = Path(c['source_receipt'])
    receipt = json.loads(receipt_path.read_text())
    if (sha(source) != c['source_sha256'] or sha(ref) != c['reference_sha256'] or
        sha(receipt_path) != c['receipt_sha256'] or receipt['timeline']['status'] != 'FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS' or
        sha(HUMANML / 'train.txt') != catalog['official_train_sha256'] or
        a.id not in set((HUMANML / 'train.txt').read_text().splitlines()) or
        sha(HUMANML / 'texts.zip') != catalog['caption_archive_sha256']):
        raise ValueError('source, receipt or official split changed')
    with zipfile.ZipFile(HUMANML / 'texts.zip') as z:
        captions = z.read('texts/' + a.id + '.txt')
    if hashlib.sha256(captions).hexdigest() != c['caption_sha256']:
        raise ValueError('original caption changed')
    with np.load(ref, allow_pickle=False) as f:
        fps = float(f['mocap_framerate']); ref_count = len(f['trans'])
    if abs(fps - c['reference_fps']) > 1e-10 or int(fps / 20) != c['sample_step']:
        raise ValueError('reference time contract changed')
    first = c['start_20'] * c['sample_step']; last = (c['stop_20'] - 1) * c['sample_step']
    frames = np.arange(first, last + 1, dtype=np.int64)
    if first < 0 or last >= ref_count or len(frames) < 2:
        raise ValueError('clip outside reference')
    with np.load(source, allow_pickle=False) as f:
        trans, orient, body = f['trans'].copy(), f['root_orient'].copy(), f['pose_body'].copy()
        betas = np.asarray(f['betas'], dtype=np.float32).reshape(-1)[:16]
        gender = str(f['gender'].item())
    if (len(trans) != ref_count or orient.shape != (ref_count,3) or body.shape != (ref_count,63) or
        betas.shape != (16,) or gender not in ('male','female','neutral')):
        raise ValueError('unsupported SMPL-X source format')
    model_path = MODEL / 'smplx' / ('SMPLX_' + gender.upper() + '.npz'); model_hash = sha(model_path)
    a.output.mkdir(parents=False, exist_ok=False)
    with contextlib.redirect_stdout(sys.stderr):
        model = smplx.create(str(MODEL), model_type='smplx', gender=gender, num_betas=16,
                             use_pca=False, batch_size=1, ext='npz').cpu()
    all_points = []
    for offset in range(0, len(frames), 60):
        ix = frames[offset:offset+60]; n = len(ix)
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        zero = lambda width: torch.zeros((n, width), dtype=torch.float32)
        with torch.no_grad():
            result = model(betas=tensor(np.broadcast_to(betas, (n,16)).copy()), global_orient=tensor(orient[ix]),
                body_pose=tensor(body[ix]), transl=tensor(trans[ix]), left_hand_pose=zero(45), right_hand_pose=zero(45),
                jaw_pose=zero(3), leye_pose=zero(3), reye_pose=zero(3), expression=zero(model.num_expression_coeffs),
                return_verts=False).joints[:, :22].numpy()
        all_points.append(result)
    points = np.concatenate(all_points)
    if points.shape != (len(frames),22,3) or not np.isfinite(points).all():
        raise ValueError('invalid human points')
    if sha(source) != c['source_sha256'] or sha(ref) != c['reference_sha256'] or sha(model_path) != model_hash:
        raise RuntimeError('inputs changed during reconstruction')
    artifact = a.output / 'human_rawfps.npz'
    with artifact.open('xb') as f:
        np.savez_compressed(f, human=points, source_frames=frames, source_times=frames.astype(np.float64) / fps)
    result = {'id': a.id, 'source_path': c['source_path'], 'source_parent': c['source_directory'],
              'source_sha256': c['source_sha256'], 'reference_sha256': c['reference_sha256'],
              'receipt_sha256': c['receipt_sha256'], 'catalog_sha256': sha(CATALOG), 'model_sha256': model_hash,
              'gender': gender, 'reference_fps': fps, 'official_stride': c['sample_step'],
              'frames': len(frames), 'first_last_raw_frames': [int(first), int(last)],
              'original_caption_records': c['captions'], 'caption_member_sha256': c['caption_sha256'],
              'official_split': 'train', 'artifact_sha256': sha(artifact), 'script_sha256': sha(Path(__file__)),
              'training_ready': False, 'limitations': ['Frame identity inherited from measured-marker receipt.',
              'Locked-head SMPL-X with zero face/hand poses; not verified equivalent to official SMPL+H.',
              'Reference FPS is from reference metadata; actual capture time and caption semantics unaccepted.',
              'Raw-rate reconstruction is experimental, without robot or physical acceptance.']}
    with (a.output / 'human_report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'id': a.id, 'frames': len(frames), 'training_ready': False}), flush=True)


if __name__ == '__main__':
    main()
