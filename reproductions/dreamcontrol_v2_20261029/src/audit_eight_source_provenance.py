"""Recheck eight candidate source hashes and declared source-group separation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
ALIGN = ROOT / 'ssm_id_alignment_20261030/ids.jsonl'
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = INPUT / 'manifest.json'
    built = json.loads((INPUT / 'build_report.json').read_text())
    if sha(manifest) != built['manifest_sha256']:
        raise ValueError('manifest changed')
    records = json.loads(manifest.read_text())['records']
    align = {x['motion_id']: x for x in map(json.loads, ALIGN.read_text().splitlines()) if not x['mirrored']}
    a = json.loads((ROOT / 'humanml_mirror_time_witness_v2_20261030/report.json').read_text())
    b = json.loads((ROOT / 'ssm_three_human_candidates_20261031/report.json').read_text())
    witnesses = {x['id']: x for x in a['witnesses'] + b['witnesses']}
    handball = json.loads((ROOT / 'bmlhandball_overlap_witness_v2_20261030/report.json').read_text())
    handball = {x['source_path']: x for x in handball['witnesses']}
    official = {}
    for split in ('train', 'val', 'test'):
        for ident in (HUMANML / (split + '.txt')).read_text().splitlines():
            if ident in official:
                raise ValueError('official ID appears in multiple splits')
            official[ident] = split
    clip_ids = {'BML_S04_Expert': '002323', 'BML_S08_Novice': '001705',
                'ACCAD_004347': '004347', 'MOSH_000798_crop': '000798'}
    cases = []; groups = {}; raw_splits = {}
    for rec in records:
        ident = rec['sample_id']; clip = clip_ids.get(ident, ident)
        if ident in ('000008', '000069', '003017', '013303'):
            info, w = align[clip], witnesses[clip]
            if info['source_path'] != rec['source_id']:
                raise ValueError('SSM source mapping changed')
            receipt_path = Path(info['source_receipt']); receipt = json.loads(receipt_path.read_text())
            path, ref = Path(receipt['smplx_file']), Path(receipt['reference_file'])
            if (sha(receipt_path) != w['source_receipt_sha256'] or sha(path) != w['source_sha256'] or
                sha(ref) != w['reference_sha256'] or receipt['timeline']['status'] != 'FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS'):
                raise ValueError('SSM receipt or files changed')
            expected = w['source_sha256']
            basis = 'prior measured-marker frame identity; model equivalence not independently established'
        elif ident.startswith('BML_'):
            w = handball[rec['source_id']]
            path = Path(w['candidate_path']); expected = w['candidate_sha256']; ref = None
            if expected not in rec['provenance_keys']['raw_candidate_sha256']:
                raise ValueError('BML candidate hash changed')
            basis = 'candidate and nominal overlap only; official intermediate identity not established'
        else:
            folder = (ROOT / 'accad_004347_fullrate_20261031' if ident == 'ACCAD_004347'
                      else ROOT / 'mosh_000798_fullrate_20261031/output')
            r = json.loads((folder / 'human_report.json').read_text())
            if r['source_path'] != rec['source_id'] and './' + r['source_path'].lstrip('./') != './' + rec['source_id'].lstrip('./'):
                raise ValueError('AMASS source mapping changed')
            path = Path(r['source_candidate_path']); expected = r['source_sha256']; ref = None
            basis = 'catalog-matched candidate; official intermediate identity not established'
        if sha(path) != expected:
            raise ValueError('source candidate bytes changed')
        if groups.setdefault(rec['group_id'], rec['split']) != rec['split']:
            raise ValueError('declared group crosses splits')
        if raw_splits.setdefault(expected, rec['split']) != rec['split']:
            raise ValueError('identical candidate file crosses splits')
        cases.append({'sample_id': ident, 'caption_id': clip, 'source_id': rec['source_id'],
                      'candidate_path': str(path), 'candidate_sha256': expected,
                      'reference_sha256': sha(ref) if ref else None, 'declared_group': rec['group_id'],
                      'auxiliary_split': rec['split'], 'official_id_split': official[clip],
                      'identity_evidence': basis, 'identity_accepted': False, 'training_ready': False})
    out = {'manifest_sha256': sha(manifest), 'alignment_sha256': sha(ALIGN),
           'official_split_sha256': {x: sha(HUMANML / (x + '.txt')) for x in ('train', 'val', 'test')},
           'cases': cases, 'declared_groups': len(groups), 'cross_split_equal_candidate_hashes': 0,
           'training_ready': False,
           'limitations': ['Matching bytes and directory names do not prove subject independence, licenses or source version.',
                           'Some auxiliary splits intentionally differ from official HumanML split.',
                           'No complete identity or physical acceptance.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'cases': len(cases), 'groups': len(groups), 'training_ready': False}))


if __name__ == '__main__':
    main()
