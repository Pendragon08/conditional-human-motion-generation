"""Select time-stamped atomic-action annotations and estimate body-archive size.

The signed URL manifest remains private and is never copied into outputs.
Selection is keyword-based as described in the v2 paper; it does not establish
semantic correctness or equivalence to the authors' undisclosed exact split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


def select(root: Path, manifest: Path, keywords: list[str]) -> tuple[dict, list[dict]]:
    dataset = json.loads(manifest.read_text(encoding='utf-8'))['sequences']
    words = [w.lower().strip() for w in keywords if w.strip()]
    if not words:
        raise ValueError('at least one keyword required')
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\w*\b', re.IGNORECASE)
    segments = []
    files_read = 0
    for seq, groups in dataset.items():
        file = root / seq / 'narration' / 'atomic_action.csv'
        if not file.is_file():
            continue
        files_read += 1
        with file.open(newline='', encoding='utf-8-sig') as source:
            reader = csv.DictReader(source)
            required = {'start_time', 'end_time', 'Describe my atomic actions'}
            if not reader.fieldnames or not required <= set(reader.fieldnames):
                raise ValueError(f'unexpected atomic-action schema in {file}')
            for row in reader:
                caption = row.get('Describe my atomic actions', '')
                matches = sorted(set(m.group(0).lower() for m in pattern.finditer(caption)))
                if not matches:
                    continue
                try:
                    start, end = float(row['start_time']), float(row['end_time'])
                except (ValueError, TypeError):
                    continue
                if not (0 <= start < end):
                    continue
                segments.append({'sequence': seq, 'start_time': start, 'end_time': end,
                                 'keywords': ';'.join(matches), 'caption': caption})
    selected = {row['sequence'] for row in segments}
    body_sizes = [dataset[s]['body_motion']['file_size_bytes'] for s in selected if 'body_motion' in dataset[s]]
    summary = {'keywords': words, 'available_atomic_annotation_files': files_read,
               'atomic_annotation_files_expected': sum('narration_atomic_action_csv' in x for x in dataset.values()),
               'selected_segments': len(segments), 'selected_sequences': len(selected),
               'body_motion_archives_available': len(body_sizes),
               'compressed_body_archives_bytes': sum(body_sizes),
               'selection_note': 'Contains keyword matches; exact paper split, temporal conversion and semantics not yet verified.'}
    return summary, segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keyword', action='append', default=[])
    args = p.parse_args()
    summary, segments = select(args.annotations, args.manifest, args.keyword or ['drawer', 'open', 'cabinet'])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'selection_summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with (args.output / 'selected_segments.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'start_time', 'end_time', 'keywords', 'caption'])
        writer.writeheader()
        writer.writerows(segments)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
