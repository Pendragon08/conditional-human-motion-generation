"""Independently compare extracted source bytes with the fixed selection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    selection_hash=sha(a.selection)
    selection=json.loads(a.selection.read_text(encoding='utf8'))
    report_path=a.source/'report.json';report_hash=sha(report_path)
    report=json.loads(report_path.read_text(encoding='utf8'))
    selected={r['id']:r for r in selection['members']}
    found={r['id']:r for r in report['members']}
    if (len(selected)!=20 or len(found)!=20 or set(selected)!=set(found)
        or report['selection_sha256']!=selection_hash or report['training_ready'] is not False):
        raise ValueError('selection or report coverage mismatch')
    counts=Counter();cases=[]
    for sid,expected in selected.items():
        row=found[sid];relative=Path(row['output'])
        target=(a.source/relative).resolve(strict=True)
        if (relative.is_absolute() or not target.is_relative_to(a.source.resolve()) or not target.is_file()
            or target.stat().st_size!=expected['member_size'] or sha(target)!=expected['member_sha256']
            or row['output_sha256']!=expected['member_sha256'] or row['member']!=expected['member']
            or row['auxiliary_split']!=expected['auxiliary_split']):
            raise ValueError('extracted file or source binding mismatch')
        counts[row['auxiliary_split']]+=1
        cases.append({'id':sid,'member':row['member'],'output':row['output'],'sha256':sha(target)})
    if sha(a.selection)!=selection_hash or sha(report_path)!=report_hash:
        raise RuntimeError('selection or report changed')
    out={'cases':cases,'auxiliary_counts':dict(counts),'selection_sha256':selection_hash,
         'extraction_report_sha256':report_hash,'script_sha256':sha(Path(__file__)),'training_ready':False,
         'limitations':['Checks extracted files against previous archive inventory; does not independently re-read tar.',
                        'Nominal source and subject names do not prove identity, actual clock or semantic alignment.',
                        'No robot conversion, caption acceptance, physical quality or training acceptance.']}
    with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'cases':len(cases),'auxiliary_counts':dict(counts),'training_ready':False}))


if __name__=='__main__':main()
