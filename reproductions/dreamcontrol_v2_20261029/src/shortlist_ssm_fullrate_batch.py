"""Find hash-verified SSM train candidates in new recording directories."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ALIGN = ROOT / 'ssm_id_alignment_20261030/ids.jsonl'
DATA = ROOT / 'five_conservative_preview_20261031/manifest.json'
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    prior = json.loads(DATA.read_text())['records']
    occupied_dirs = {x for rec in prior for x in rec.get('provenance_keys', {}).get('source_directory', [])}
    train = set((HUMANML / 'train.txt').read_text().splitlines())
    rows = [json.loads(line) for line in ALIGN.read_text().splitlines()]
    candidates = []; skipped = {}; skipped_details = []
    with zipfile.ZipFile(HUMANML / 'texts.zip') as texts:
        for row in rows:
            if row['mirrored'] or row['motion_id'] not in train:
                continue
            parent = str(Path(row['source_path']).parent).lstrip('./')
            if parent in occupied_dirs:
                skipped['existing_directory'] = skipped.get('existing_directory', 0) + 1
                continue
            rate = row.get('recorded_effective_fps')
            ref_fps = row.get('reference_fps')
            rate_supported = (isinstance(rate, (int, float)) and min(abs(rate - 20.), abs(rate - 30.)) < 0.001 and
                              isinstance(ref_fps, (int, float)) and min(abs(ref_fps - 60.), abs(ref_fps - 120.)) < 0.001)
            if not (rate_supported and 50 <= row['stop'] - row['start'] <= 200):
                skipped['unsupported_rate_or_length'] = skipped.get('unsupported_rate_or_length', 0) + 1
                skipped_details.append({'id': row['motion_id'], 'source_directory': parent,
                                        'reference_fps': row.get('reference_fps'), 'effective_fps': rate,
                                        'sample_stride': row.get('recorded_stride'),
                                        'clip_samples': row['stop']-row['start']})
                continue
            receipt_path = Path(row['source_receipt']); receipt = json.loads(receipt_path.read_text())
            source, reference = Path(receipt['smplx_file']), Path(receipt['reference_file'])
            if (sha(source) != receipt['smplx_sha256'] or sha(reference) != receipt['reference_sha256'] or
                receipt['timeline']['status'] != 'FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS'):
                raise ValueError('receipt or source identity evidence changed')
            raw = texts.read('texts/' + row['motion_id'] + '.txt')
            descriptions = [line.split('#') for line in raw.decode('utf8').splitlines()]
            if any(len(x) != 4 for x in descriptions):
                raise ValueError('malformed caption')
            candidates.append({'id': row['motion_id'], 'source_path': row['source_path'], 'source_directory': parent,
                'source_receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path),
                'source': str(source), 'source_sha256': sha(source), 'reference': str(reference),
                'reference_sha256': sha(reference), 'reference_fps': row['reference_fps'],
                'sample_step': row['recorded_stride'], 'start_20': row['start'], 'stop_20': row['stop'],
                'clip_samples_20': row['stop']-row['start'],
                'captions': [{'text': x[0], 'tags': x[2:]} for x in descriptions],
                'caption_sha256': hashlib.sha256(raw).hexdigest(), 'training_ready': False})
    out = {'alignment_sha256': sha(ALIGN), 'prior_manifest_sha256': sha(DATA),
           'official_train_sha256': sha(HUMANML / 'train.txt'), 'caption_archive_sha256': sha(HUMANML / 'texts.zip'),
           'occupied_source_directories': sorted(occupied_dirs), 'candidates': candidates, 'skipped': skipped,
           'unsupported_details': skipped_details,
           'training_ready': False,
           'limitations': ['Marker frame identity inherited; not new independent frame truth or source subject independence.',
                           'Caption semantics, source model equivalence and physical quality unaccepted.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    print(json.dumps({'candidates': len(candidates), 'directories': len({x['source_directory'] for x in candidates}), 'training_ready': False}))


if __name__ == '__main__':
    main()
