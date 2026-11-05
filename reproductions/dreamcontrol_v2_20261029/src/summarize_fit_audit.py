"""Validate and summarize completed per-clip batch_fit_audit outputs.

Clip-level means weight clips equally. Numbers are diagnostics, not pass/fail
thresholds or evidence of dynamics, collision, or paper-level reproduction.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    done = json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    samples = manifest['samples']
    if done.get('status') != 'finished' or done.get('record_count') != len(samples):
        raise ValueError('audit completion metadata is inconsistent')
    found, errors = [], []
    identifiers = set()
    for sample in samples:
        pair = (sample['archive'], sample['member'])
        if pair in identifiers:
            raise ValueError('duplicate source member in manifest')
        identifiers.add(pair)
        key = hashlib.sha256((pair[0]+'\0'+pair[1]).encode('utf-8')).hexdigest()
        item = json.loads((directory/(key+'.json')).read_text(encoding='utf-8'))
        if (item.get('archive'), item.get('member')) != pair:
            raise ValueError('record source does not match manifest')
        if item.get('status') == 'ok':
            found.append(item)
        elif item.get('status') == 'error':
            errors.append({'archive': pair[0], 'member': pair[1], 'error': item.get('error')})
        else:
            raise ValueError('unknown record status')
    keys = ('mean_keypoint_error_m', 'max_keypoint_error_m', 'mean_foot_error_m', 'max_foot_error_m',
            'mean_wrist_error_m', 'contact_proxy_velocity_error_mps', 'max_root_translation_m', 'max_root_rotation_rad')
    methods = {}
    for method in manifest['config']['methods']:
        metrics = {}
        for key in keys:
            numbers = [x['methods'][method][key] for x in found if x['methods'][method][key] is not None]
            values = np.asarray(numbers, dtype=np.float64)
            if values.size and (not np.isfinite(values).all() or (values < 0).any()):
                raise ValueError(f'invalid metric: {method}/{key}')
            metrics[key] = {'n': int(values.size), 'mean': float(values.mean()) if values.size else None,
                            'p95': float(np.quantile(values, .95)) if values.size else None,
                            'max': float(values.max()) if values.size else None}
        methods[method] = metrics
    return {'expected': len(samples), 'successful': len(found), 'failed': len(errors), 'errors': errors,
            'source_counts': dict(Counter(x['archive'] for x in found)), 'methods': methods,
            'note': 'Nonrandom clip-level diagnostics; this does not establish physical validity or paper reproduction.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.audit_dir)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    print(json.dumps({'expected': report['expected'], 'successful': report['successful'], 'failed': report['failed']}))


if __name__ == '__main__':
    main()
