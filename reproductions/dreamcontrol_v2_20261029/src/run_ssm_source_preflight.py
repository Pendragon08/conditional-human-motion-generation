"""Recompute marker evidence for the two SSM samples in the reviewed preview."""
import hashlib
import json
from pathlib import Path
from audit_ssm_source_evidence import audit_ssm_source

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PRE=ROOT/'four_semantic_subset_20261101'
ALIGN=ROOT/'ssm_id_alignment_20261030/ids.jsonl'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=json.loads((PRE/'audit.json').read_text())
    assert sha(PRE/'manifest.json')==report['manifest_sha256']
    records={r['sample_id']:r for r in json.loads((PRE/'manifest.json').read_text())['records']}
    align={r['motion_id']:r for r in map(json.loads,ALIGN.read_text().splitlines()) if not r['mirrored']}
    witness=json.loads((ROOT/'humanml_mirror_time_witness_v2_20261030/report.json').read_text())
    first=next(x for x in witness['witnesses'] if x['id']=='000008')
    second=json.loads((ROOT/'ssm_rawfps_human_batch_20261031/011120/human_report.json').read_text())
    expected={'000008':first['source_receipt_sha256'],'011120':second['receipt_sha256']}
    cases=[]
    for sid in ('000008','011120'):
        info=align[sid]
        cases.append(audit_ssm_source(records[sid],info,receipt_path=info['source_receipt'],receipt_sha256=expected[sid]))
    out={'manifest_sha256':sha(PRE/'manifest.json'),'alignment_sha256':sha(ALIGN),'cases':cases,
         'other_samples_without_this_ssm_evidence':['ACCAD_004347','MOSH_000798_crop'],
         'training_ready':False,'script_sha256':sha(Path(__file__)),'module_sha256':sha(HERE/'audit_ssm_source_evidence.py'),
         'limitations':['This supplies only marker correspondence; all source/time/license/physics conditions remain separate.',
                        'Does not independently reconstruct official HumanML intermediate.']}
    with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'cases':len(cases),'mismatched_header_rates':sum(x['header_rates_disagree'] for x in cases),'training_ready':False}))


if __name__=='__main__':main()
