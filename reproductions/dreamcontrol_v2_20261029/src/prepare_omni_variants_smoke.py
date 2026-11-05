"""Snapshot the first six completed variant outputs for an early audit."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_variants_gpu_20261101'
HERE=Path(__file__).resolve().parent
ROBOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    with (SOURCE/'records.jsonl').open() as f:
        lines=[f.readline() for _ in range(6)]
    if len(lines)!=6 or not all(lines):raise ValueError('six complete lines not yet available')
    rows=[json.loads(s) for s in lines]
    if len({r['sample_id'] for r in rows})!=6 or any(r['technical_processing']!='success' for r in rows):
        raise ValueError('first six records not all successful')
    for row in rows:
        for key in ('state30','state20','feature'):
            src=(SOURCE/row[key]).resolve(strict=True)
            if not src.is_relative_to(SOURCE.resolve()) or sha(src)!=row[key+'_sha256']:
                raise ValueError('snapshot source hash mismatch')
            with src.open('rb') as a,(HERE/row[key]).open('xb') as b:shutil.copyfileobj(a,b)
            if sha(HERE/row[key])!=row[key+'_sha256']:raise ValueError('snapshot copy mismatch')
    with (HERE/'records.jsonl').open('x') as f:f.writelines(lines)
    report={'source_archive_sha256':sha(ROOT/'data/OmniRetarget_Dataset/robot-object.zip'),
            'urdf29_sha256':sha(ROBOT/'g1_29dof.urdf'),'urdf27_sha256':sha(ROBOT/'g1_27dof.urdf'),
            'records_sha256':sha(HERE/'records.jsonl'),'selected':6,'attempted':6,
            'source_queue':'omni_variants_gpu_20261101','scope':'snapshot of first six completed records, not whole batch',
            'training_ready':False}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
