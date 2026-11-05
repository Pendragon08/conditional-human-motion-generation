"""Reconstruct an explicitly indexed AMASS candidate on CPU before retargeting.

Index mapping and FPS are conditional on source metadata and published
HumanML preprocessing, not verified capture times or official body identity.
"""
import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import smplx
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--catalog-sha256', required=True)
    ap.add_argument('--id', required=True)
    ap.add_argument('--model-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    catalog_hash = sha(a.catalog)
    if catalog_hash != a.catalog_sha256: raise ValueError('candidate catalog hash mismatch')
    catalog = json.loads(a.catalog.read_text())
    matches = [r for r in catalog['selected'] if r['motion_id'] == a.id]
    if len(matches) != 1: raise ValueError('candidate ID must be unique')
    item = matches[0]; source = Path(item['candidate_path']).resolve(strict=True)
    source_hash = sha(source)
    if source_hash != item['candidate_sha256']: raise ValueError('source hash mismatch')
    with np.load(source, allow_pickle=False) as data:
        pose = data['poses'].copy(); trans = data['trans'].copy(); betas = data['betas'].copy()
        fpskey = next((k for k in ('mocap_framerate', 'mocap_frame_rate') if k in data.files), None)
        if fpskey is None: raise ValueError('missing source frame rate')
        fps = float(data[fpskey].item()); gender = str(data['gender'].item())
    if (fps != 120 or pose.ndim != 2 or pose.shape[1] != 165 or trans.shape != (len(pose), 3)
        or betas.shape != (16,) or gender not in ('male', 'female', 'neutral')):
        raise ValueError('unsupported source format or declared FPS')
    start, end = int(item['index_start_20']), int(item['index_end_20'])
    frames = np.arange(start * 6, (end - 1) * 6 + 1, dtype=np.int64)
    if start < 0 or end <= start or len(frames) < 2 or frames[-1] >= len(pose):
        raise ValueError('indexed clip exceeds candidate')
    model_file = a.model_root / 'smplx' / ('SMPLX_' + gender.upper() + '.npz')
    model_hash = sha(model_file)
    a.output.mkdir(parents=False, exist_ok=False)
    with contextlib.redirect_stdout(sys.stderr):
        model = smplx.create(str(a.model_root), model_type='smplx', gender=gender, num_betas=16,
                             use_pca=False, batch_size=1, ext='npz').cpu()
    chunks = []
    for offset in range(0, len(frames), 60):
        ix = frames[offset:offset + 60]; n = len(ix)
        conv = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        zero = lambda width: torch.zeros((n, width), dtype=torch.float32)
        with torch.no_grad():
            points = model(betas=conv(np.broadcast_to(betas, (n, 16)).copy()),
                           global_orient=conv(pose[ix, :3]), body_pose=conv(pose[ix, 3:66]),
                           transl=conv(trans[ix]), left_hand_pose=zero(45), right_hand_pose=zero(45),
                           jaw_pose=zero(3), leye_pose=zero(3), reye_pose=zero(3),
                           expression=zero(model.num_expression_coeffs), return_verts=False).joints[:, :22].numpy()
        chunks.append(points)
    human = np.concatenate(chunks)
    if human.shape != (len(frames), 22, 3) or not np.isfinite(human).all():
        raise ValueError('invalid human reconstruction')
    if sha(source) != source_hash or sha(a.catalog) != catalog_hash or sha(model_file) != model_hash:
        raise RuntimeError('input changed during reconstruction')
    output = a.output / 'human_120hz.npz'
    with output.open('xb') as f:
        np.savez_compressed(f, human=human, source_frames=frames, source_times=frames.astype(np.float64) / fps)
    report = {'id': a.id, 'source_path': item['source_path'], 'source_parent': item['parent_group'],
              'source_candidate_path': str(source), 'source_sha256': source_hash,
              'catalog_sha256': catalog_hash, 'model_sha256': model_hash, 'gender': gender,
              'fps_declared': fps, 'frames': len(frames), 'first_last_raw_frames': [int(frames[0]), int(frames[-1])],
              'original_caption_records': item['raw_captions'], 'caption_member_sha256': item['caption_member_sha256'],
              'official_split': item['official_split'], 'artifact_sha256': sha(output),
              'script_sha256': sha(Path(__file__)), 'training_ready': False,
              'limitations': ['SMPL-X locked-head approximation; not independently verified SMPL+H.',
                              'Candidate identity, real FPS and caption semantics are not established.',
                              'No robot retargeting, contact or physical quality acceptance.']}
    with (a.output / 'human_report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    print(json.dumps({'id': a.id, 'frames': len(frames), 'training_ready': False}), flush=True)


if __name__ == '__main__': main()
