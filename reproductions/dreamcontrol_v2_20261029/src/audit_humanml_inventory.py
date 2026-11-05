"""Read-only inventory of the existing HumanML annotations and AMASS source files."""
import collections
import csv
import json
from pathlib import Path
import zipfile

BASE = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass')
PUBLIC = BASE / 'humanml_pairing_20261021/code/public'
DATA = PUBLIC / 'HumanML3D'


def main():
    with zipfile.ZipFile(DATA / 'texts.zip') as z:
        entries = [x for x in z.infolist() if not x.is_dir()]
        bad = z.testzip()
        txts = [Path(x.filename).stem for x in entries if x.filename.endswith('.txt')]
        paths = [x.filename for x in entries[:4]]
    all_ids = set((DATA / 'all.txt').read_text().splitlines())
    splits = {s: set((DATA / (s + '.txt')).read_text().splitlines()) for s in ('train', 'val', 'test')}
    counts = collections.Counter()
    row_count = 0
    with (PUBLIC / 'index.csv').open(newline='', encoding='utf8') as f:
        for row in csv.DictReader(f):
            parts = Path(row['source_path']).parts
            if len(parts) > 1:
                counts[parts[1]] += 1
            row_count += 1
    raw = []
    for folder in ('remaining_20261021','remaining_20261021_v4','remaining_20261021_v5','recovery_20261021_v1'):
        for verification in sorted((BASE / folder).glob('*/server_verification.json')):
            record = json.loads(verification.read_text())
            raw.append({'subset': record.get('subset'), 'status': record.get('status'),
                        'manifest_files': len(record.get('files', [])), 'directory': str(verification.parent)})
    text_ids = set(txts)
    print(json.dumps({'text_zip_entries': len(entries), 'text_txt_files': len(txts),
        'text_unique_ids': len(text_ids), 'zip_crc_bad': bad, 'zip_sample_paths': paths,
        'all_ids': len(all_ids), 'split_sizes': {k:len(v) for k,v in splits.items()},
        'split_union_size': len(set.union(*splits.values())),
        'split_overlaps': {a+'_'+b:len(splits[a] & splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
        'all_missing_text_ids': len(all_ids - text_ids), 'extra_text_ids': len(text_ids - all_ids),
        'index_rows': row_count, 'index_source_counts': dict(counts), 'raw_amass_manifest': raw,
        'warning': 'Inventory and archive CRC only; no raw source mapping, individual present-file hashes, license or training-data acceptance.'}, indent=2))


if __name__ == '__main__':
    main()
