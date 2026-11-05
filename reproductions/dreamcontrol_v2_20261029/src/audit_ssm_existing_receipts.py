"""Check prior SSM alignment receipts against current raw inputs, read-only."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_batch_v1')
OUT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/ssm_receipt_audit_20261030')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for p in sorted((ROOT / 'source_receipts').glob('*.json')):
        doc = json.loads(p.read_text())
        ref, target = Path(doc['reference_file']), Path(doc['smplx_file'])
        timeline = doc['timeline']
        fps = float(timeline['reference_fps'])
        stride = int(timeline['reference_stride'])
        rows.append({'source_group': doc['source_group'], 'receipt': str(p),
            'receipt_sha256': digest(p), 'ref_hash_matches': ref.is_file() and digest(ref) == doc['reference_sha256'],
            'target_hash_matches': target.is_file() and digest(target) == doc['smplx_sha256'],
            'marker_status': timeline['status'], 'marker_max_abs_m': timeline['marker_max_abs_m'],
            'reference_fps': fps, 'target_header_fps': timeline['smplx_header_fps'],
            'stride': stride, 'effective_fps': fps / stride,
            'within_0_01_hz_of_20': abs(fps / stride - 20) <= 0.01})
    accepted = json.loads((ROOT / 'accepted.json').read_text())
    summary = {'receipts':len(rows), 'reference_files_hash_match':sum(r['ref_hash_matches'] for r in rows),
        'target_files_hash_match':sum(r['target_hash_matches'] for r in rows),
        'timeline_status_counts':dict(collections.Counter(r['marker_status'] for r in rows)),
        'stride_counts':dict(collections.Counter(str(r['stride']) for r in rows)),
        'target_header_fps_counts':dict(collections.Counter(str(r['target_header_fps']) for r in rows)),
        'effective_fps_min_max':[min(r['effective_fps'] for r in rows),max(r['effective_fps'] for r in rows)],
        'not_near_20hz':sum(not r['within_0_01_hz_of_20'] for r in rows),
        'accepted_segments':len(accepted), 'accepted_ids':len({r['motion_id'] for r in accepted}),
        'accepted_source_groups':len({r['source_group'] for r in accepted}),
        'accepted_mirrored_segments':sum(r['mirrored'] for r in accepted),
        'dynamics_validated':False,
        'limitation':'Existing receipts and input hashes rechecked; measured-marker comparisons not rerun here, G1 output is prior 29-DoF/262D approximation, no v2 acceptance.'}
    OUT.mkdir(mode=0o700)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    with (OUT / 'rows.jsonl').open('x', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
