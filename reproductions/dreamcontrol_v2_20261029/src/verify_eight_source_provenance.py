"""Re-read source bytes and official splits without the source audit module."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    manifest = INPUT / 'manifest.json'
    if sha(manifest) != report['manifest_sha256']:
        raise ValueError('manifest changed')
    records = json.loads(manifest.read_text())['records']
    if len(records) != len(report['cases']) or len(records) != 8:
        raise ValueError('case coverage mismatch')
    splits = {s: set((HUMANML / (s + '.txt')).read_text().splitlines()) for s in ('train', 'val', 'test')}
    for split in splits:
        if sha(HUMANML / (split + '.txt')) != report['official_split_sha256'][split]:
            raise ValueError('official split changed')
    seen = {}; groups = {}; differences = []
    for r, c in zip(records, report['cases']):
        if (r['sample_id'] != c['sample_id'] or r['source_id'] != c['source_id'] or
            r['group_id'] != c['declared_group'] or r['split'] != c['auxiliary_split']):
            raise ValueError('source mapping changed')
        actual = sha(Path(c['candidate_path']))
        if actual != c['candidate_sha256'] or seen.setdefault(actual, r['split']) != r['split']:
            raise ValueError('source hash changed or crosses splits')
        if groups.setdefault(r['group_id'], r['split']) != r['split']:
            raise ValueError('declared group crosses splits')
        matching = [name for name, ids in splits.items() if c['caption_id'] in ids]
        if matching != [c['official_id_split']]:
            raise ValueError('official ID split mismatch')
        if c['official_id_split'] != r['split']:
            differences.append({'sample_id': r['sample_id'], 'official': c['official_id_split'], 'auxiliary': r['split']})
        if c['identity_accepted'] or c['training_ready']:
            raise ValueError('unverified identity marked accepted')
    out = {'report_sha256': sha(HERE / 'report.json'), 'manifest_sha256': sha(manifest),
           'cases_rechecked': len(records), 'candidate_hashes': len(seen), 'declared_groups': len(groups),
           'official_vs_auxiliary_split_differences': differences, 'training_ready': False,
           'limitations': 'Same candidate files, not independent source identity or subject, license, real FPS or physical acceptance.'}
    with (HERE / 'independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
