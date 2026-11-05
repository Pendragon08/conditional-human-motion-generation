"""Independently recheck SSM provenance/timeline and selected same-model FK."""
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


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    torch.set_num_threads(1); folder = args.output; report = json.loads((folder/'human_report.json').read_text())
    assert sha(CATALOG) == report['catalog_sha256']
    c = next(x for x in json.loads(CATALOG.read_text())['candidates'] if x['id'] == report['id'])
    source, reference = Path(c['source']), Path(c['reference'])
    assert sha(source) == c['source_sha256'] == report['source_sha256']
    assert sha(reference) == c['reference_sha256'] == report['reference_sha256']
    assert sha(Path(c['source_receipt'])) == c['receipt_sha256'] == report['receipt_sha256']
    assert sha(folder/'human_rawfps.npz') == report['artifact_sha256']
    model_path = MODEL/'smplx'/('SMPLX_' + report['gender'].upper() + '.npz')
    assert sha(model_path) == report['model_sha256']
    with zipfile.ZipFile(HUMANML/'texts.zip') as z:
        caption = z.read('texts/' + report['id'] + '.txt')
    assert hashlib.sha256(caption).hexdigest() == report['caption_member_sha256']
    with np.load(reference,allow_pickle=False) as f:
        fps = float(f['mocap_framerate']); count = len(f['trans'])
    with np.load(folder/'human_rawfps.npz',allow_pickle=False) as f:
        indices, times, points = f['source_frames'].copy(), f['source_times'].copy(), f['human'].copy()
    first = c['start_20'] * c['sample_step']; last = (c['stop_20'] - 1) * c['sample_step']
    np.testing.assert_array_equal(indices, np.arange(first,last+1))
    np.testing.assert_allclose(times, indices.astype(float)/fps, atol=1e-12, rtol=0)
    assert points.shape == (last-first+1,22,3) and last < count and np.isfinite(points).all()
    selected = np.array([0, len(indices)//2, len(indices)-1]); raw = indices[selected]
    with np.load(source,allow_pickle=False) as f:
        body, orient, trans = f['pose_body'][raw], f['root_orient'][raw], f['trans'][raw]
        betas = np.asarray(f['betas'], dtype=np.float32).reshape(-1)[:16]
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    zero = lambda width: torch.zeros((3,width),dtype=torch.float32)
    with contextlib.redirect_stdout(sys.stderr):
        model = smplx.create(str(MODEL),model_type='smplx',gender=report['gender'],num_betas=16,
                             use_pca=False,batch_size=3,ext='npz').cpu()
        with torch.no_grad():
            out = model(betas=tensor(np.broadcast_to(betas,(3,16)).copy()), global_orient=tensor(orient),
                        body_pose=tensor(body),transl=tensor(trans), left_hand_pose=zero(45),right_hand_pose=zero(45),
                        jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),
                        return_verts=False).joints[:,:22].numpy()
    err = float(np.max(np.abs(out-points[selected])))
    assert err < 1e-5
    result = {'human_report_sha256': sha(folder/'human_report.json'), 'artifact_sha256': sha(folder/'human_rawfps.npz'),
              'source_sha256': sha(source), 'reference_sha256': sha(reference),
              'frames_verified': len(indices), 'first_middle_last_raw_frames': raw.tolist(),
              'same_model_max_abs_error_m': err, 'training_ready': False,
              'limitations': 'Same model library at 3 frames; does not establish true FPS, official body equivalence, caption semantics or physical quality.'}
    with (folder/'human_audit.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'id': report['id'], 'frames_verified': len(indices), 'training_ready': False}))


if __name__ == '__main__': main()
