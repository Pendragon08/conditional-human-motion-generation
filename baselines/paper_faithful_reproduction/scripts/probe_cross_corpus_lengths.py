"""Read-only lengths of same-name public-release clips; never assume alignment.

Only load a trusted, previously acquired official pickle. This does not infer
physical FPS from frame counts and cannot prove an upstream data defect.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    inputs = {str(p.resolve()): sha(p) for p in [a.manifest, a.reactive_pickle, a.official_contract]}
    m = json.loads(a.manifest.read_text(encoding='utf-8'))
    with a.reactive_pickle.open('rb') as stream:
        corpus = pickle.load(stream)
    rows = []
    for split, entries in m['splits'].items():
        for e in entries:
            item = corpus['data_dict'].get(e['motion_id'])
            if item is None:
                continue
            rows.append({'id': e['motion_id'], 'interval': e['interval'], 'split': split,
                'paired_frames_at_declared_20hz': e['frames'], 'released_36d_frames': int(item['length']),
                'released_shape': list(item['motion'].shape),
                'interval_identity_confirmed': False, 'eligible_for_automatic_transfer': False})
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('Inputs changed')
    result = {'status': 'READ_ONLY_LENGTH_PROBE_COMPLETED', 'script_sha256': sha(__file__),
        'inputs': inputs, 'inputs_unchanged': True,
        'unique_common_ids': len({r['id'] for r in rows}), 'common_entries': len(rows),
        'rows': rows, 'training_pairs_created': 0, 'motion_arrays_exported': False,
        'scope': 'ID match alone does not establish common raw recording interval or physical time. '
        'Released data contract says 60Hz; this probe does not independently certify that clock. '
        'Different crops, retargeting and preprocessing may explain length differences. '
        'No inference that official data are wrong, no change to ongoing ReactiveBFM training.'}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'unique_common_ids': result['unique_common_ids'], 'common_entries': len(rows)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'reactive-pickle', 'official-contract', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
