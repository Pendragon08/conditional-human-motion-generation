"""Snapshot completed queue record lines, excluding the six audited earlier."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_variants_gpu_20261101';SMOKE=ROOT/'omni_variants_smoke_20261101';HERE=Path(__file__).resolve().parent
ROBOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    smoke_report=json.loads((SMOKE/'report.json').read_text());smoke_audit=json.loads((SMOKE/'audit_v2.json').read_text())
    if sha(SMOKE/'records.jsonl')!=smoke_report['records_sha256'] or sha(SMOKE/'report.json')!=smoke_audit['report_sha256'] or smoke_audit['technical_passed']!=6:
        raise ValueError('existing six-record evidence changed')
    with (SOURCE/'records.jsonl').open('rb') as f:captured=f.read()
    complete=[line for line in captured.splitlines(keepends=True) if line.endswith(b'\n')]
    rows=[json.loads(line) for line in complete]
    prior=[json.loads(line) for line in (SMOKE/'records.jsonl').read_text().splitlines()]
    if len(rows)<7 or rows[:6]!=prior or len({x['sample_id'] for x in rows})!=len(rows):
        raise ValueError('source prefix missing, altered or duplicated')
    selected=complete[6:]
    with (HERE/'records.jsonl').open('xb') as f:f.writelines(selected)
    report={'source_archive_sha256':sha(ROOT/'data/OmniRetarget_Dataset/robot-object.zip'),
            'source_script_sha256':sha(SOURCE/'omni_variants_gpu_batch.py'),
            'urdf29_sha256':sha(ROBOT/'g1_29dof.urdf'),'urdf27_sha256':sha(ROBOT/'g1_27dof.urdf'),
            'records_sha256':sha(HERE/'records.jsonl'),'selected':len(selected),'attempted':len(selected),
            'earlier_smoke_audit_sha256':sha(SMOKE/'audit_v2.json'),
            'source_queue':'omni_variants_gpu_20261101','prefix_excluded':6,
            'scope':'completed queue prefix after first six; queue may continue; no training acceptance',
            'training_ready':False}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
