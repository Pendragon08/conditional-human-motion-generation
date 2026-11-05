"""Independently verify the raw joint-angle differences for all variants."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    report=json.loads((HERE/'report.json').read_text())
    if sha(HERE/'records.jsonl')!=report['records_sha256'] or sha(ARCHIVE)!=report['archive_sha256']:
        raise ValueError('source pin mismatch')
    lines=[json.loads(line) for line in (HERE/'records.jsonl').read_text().splitlines()]
    if len(lines)!=1514 or len({x['variant_member'] for x in lines})!=1514:
        raise ValueError('invalid variant coverage')
    seen={};smallest=float('inf');largest=0.;difference=0.
    with zipfile.ZipFile(ARCHIVE) as z:
        for row in lines:
            if row['original_member'] not in seen:
                data=z.read(row['original_member'])
                if hashlib.sha256(data).hexdigest()!=row['original_sha256']:raise ValueError('original hash')
                with np.load(io.BytesIO(data),allow_pickle=False) as f:seen[row['original_member']]=f['qpos'][:,7:36].copy()
            data=z.read(row['variant_member'])
            if hashlib.sha256(data).hexdigest()!=row['variant_sha256']:raise ValueError('variant hash')
            with np.load(io.BytesIO(data),allow_pickle=False) as f:q=f['qpos'][:,7:36].copy()
            orig=seen[row['original_member']]
            if q.shape!=orig.shape:raise ValueError('unexpected shape')
            measured=float(np.max(np.abs(q.astype(np.float64)-orig.astype(np.float64))))
            reported=row['comparison']['metrics']['joint_max_abs_rad']
            difference=max(difference,abs(measured-reported))
            if abs(measured-reported)>1e-12 or measured<=1e-5:raise ValueError('joint difference not confirmed')
            smallest=min(smallest,measured);largest=max(largest,measured)
    # The scan has 312 original keys, but one has no augmented variant.
    if len(seen)!=311 or report['source_groups']!=312 or sha(ARCHIVE)!=report['archive_sha256'] or sha(HERE/'records.jsonl')!=report['records_sha256']:
        raise RuntimeError('input changed or coverage incomplete')
    out={'scan_report_sha256':sha(HERE/'report.json'),'scan_records_sha256':report['records_sha256'],
         'script_sha256':sha(Path(__file__)),'groups':len(seen),'variants':len(lines),
         'smallest_per_variant_max_joint_difference_rad':smallest,'largest_per_variant_max_joint_difference_rad':largest,
         'largest_difference_from_reported_rad':difference,'training_ready':False,
         'limitations':['Rechecks joint positions only; does not audit rigid transform calculations.',
                        'Joint differences do not establish variant quality, semantics or retargeter behavior.']}
    with (HERE/'independent_joint_audit_v2.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
