"""Check AMASS reconstruction provenance, full timeline, and three body frames."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import smplx
import torch


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ('catalog', 'result', 'model_root', 'caption_zip', 'train_split', 'output'):
        ap.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    a = ap.parse_args(); torch.set_num_threads(1)
    report_path = a.result / 'human_report.json'; report = json.loads(report_path.read_text())
    artifact = a.result / 'human_120hz.npz'; source = Path(report['source_candidate_path'])
    assert sha(a.catalog) == report['catalog_sha256'] and sha(source) == report['source_sha256']
    assert sha(artifact) == report['artifact_sha256']
    model_path = a.model_root / 'smplx' / ('SMPLX_' + report['gender'].upper() + '.npz')
    assert sha(model_path) == report['model_sha256']
    candidates = [r for r in json.loads(a.catalog.read_text())['selected'] if r['motion_id'] == report['id']]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate['candidate_sha256'] == report['source_sha256']
    assert candidate['source_path'] == report['source_path'] and candidate['raw_captions'] == report['original_caption_records']
    assert report['id'] in a.train_split.read_text().splitlines() and candidate['official_split'] == 'train'
    with zipfile.ZipFile(a.caption_zip) as archive: raw = archive.read('texts/' + report['id'] + '.txt')
    assert hashlib.sha256(raw).hexdigest() == report['caption_member_sha256']
    lines = [line.split('#') for line in raw.decode('utf8').splitlines()]
    assert [(x[0], x[2], x[3]) for x in lines] == [(x['caption'], x['start_tag'], x['end_tag']) for x in report['original_caption_records']]
    with np.load(source, allow_pickle=False) as d:
        poses, trans, betas = d['poses'].copy(), d['trans'].copy(), d['betas'].copy()
    with np.load(artifact, allow_pickle=False) as d:
        points, frames, times = d['human'].copy(), d['source_frames'].copy(), d['source_times'].copy()
    start = int(candidate['index_start_20']) * 6; stop = (int(candidate['index_end_20']) - 1) * 6
    assert points.shape == (stop - start + 1, 22, 3)
    assert np.array_equal(frames, np.arange(start, stop + 1))
    assert np.array_equal(times, frames / 120.0) and np.isfinite(points).all()
    model = smplx.create(str(a.model_root), model_type='smplx', gender=report['gender'], num_betas=16,
                         use_pca=False, batch_size=1, ext='npz').cpu()
    errors = []
    for index in (0, len(frames) // 2, len(frames) - 1):
        frame = int(frames[index]); tensor = lambda x: torch.tensor(np.asarray(x)[None], dtype=torch.float32)
        z = lambda n: torch.zeros((1, n), dtype=torch.float32)
        with torch.no_grad():
            result = model(betas=tensor(betas), global_orient=tensor(poses[frame, :3]),
                           body_pose=tensor(poses[frame, 3:66]), transl=tensor(trans[frame]),
                           left_hand_pose=z(45), right_hand_pose=z(45), jaw_pose=z(3),
                           leye_pose=z(3), reye_pose=z(3), expression=z(model.num_expression_coeffs),
                           return_verts=False).joints[0, :22].numpy()
        errors.append(float(np.max(np.abs(points[index] - result))))
    assert max(errors) < 1e-5
    result = {'human_report_sha256': sha(report_path), 'source_sha256': sha(source),
              'artifact_sha256': sha(artifact), 'sampled_frame_max_coordinate_errors_m': errors,
              'timeline_and_captions_match': True, 'training_ready': False,
              'limitations': ['Uses the same SMPL-X implementation, only three frames recomputed.',
                              'Source identity, true frame rate, semantics and physical quality remain unverified.']}
    with a.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'id': report['id'], 'frames': len(frames), 'max_error_m': max(errors)}))


if __name__ == '__main__': main()
