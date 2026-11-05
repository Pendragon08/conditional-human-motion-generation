"""Classify unexamined failures from the completed variant technical audit.

No trajectory is modified or accepted; threshold values reproduce the prior
technical preflight and are not a paper or physical acceptance standard.
"""
import collections
import hashlib
import json
from pathlib import Path
import traceback
from audit_robot_states import audit_robot_states

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_variants_gpu_20261101'
PREVIOUS=ROOT/'omni_variants_snapshot_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def load_audit(path):
    report=json.loads((path/'report.json').read_text()); audit=json.loads((path/'audit_v2.json').read_text())
    if sha(path/'report.json')!=audit['report_sha256'] or sha(path/'records.jsonl')!=report['records_sha256'] or sha(path/'audit_rows_v2.jsonl')!=audit['audit_rows_sha256']:
        raise ValueError('batch audit provenance mismatch')
    records={r['sample_id']:r for r in map(json.loads,(path/'records.jsonl').read_text().splitlines())}
    rows=list(map(json.loads,(path/'audit_rows_v2.jsonl').read_text().splitlines()))
    if len(rows)!=len(records) or {r['sample_id'] for r in rows}!=set(records):raise ValueError('audit coverage mismatch')
    failed={r['sample_id'] for r in rows if r['status']!='technical_checks_passed'}
    if failed!=set(audit['failed']):raise ValueError('failure summary mismatch')
    return report,audit,records,failed


def main():
    report,audit,records,failed=load_audit(SOURCE)
    prior_report,prior_audit,prior,prior_failed=load_audit(PREVIOUS)
    if sha(URDF)!=report['urdf27_sha256'] or not prior_failed<=failed:raise ValueError('prior failure or URDF mismatch')
    for sid,r in prior.items():
        if sid not in records or any(records[sid][k]!=r[k] for k in ('state20_sha256','feature_sha256','source_sha256')):
            raise ValueError('prior and full inputs differ')
    candidates=sorted(failed-prior_failed);counts=collections.Counter()
    with (HERE/'records.jsonl').open('x',encoding='utf8',buffering=1) as out:
        for sid in candidates:
            r=records[sid]
            try:
                check=audit_robot_states([{'sample_id':sid,'feature_path':r['feature'],'feature_sha256':r['feature_sha256'],
                    'state_path':r['state20'],'state_sha256':r['state20_sha256'],'foot_velocity_squared_threshold':.002}],
                    feature_root=SOURCE,state_root=SOURCE,urdf_path=URDF,tolerance_rad=1e-6,
                    fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
                category='joint_limit_failure' if not check['state_technical_checks_passed'] else 'unexplained_original_audit_failure'
                row={'sample_id':sid,'category':category,'state_check':check,'training_ready':False}
            except Exception as e:
                category='encoding_fk_mismatch' if isinstance(e,ValueError) and str(e).startswith('state/feature FK consistency failed:') else 'other_error'
                row={'sample_id':sid,'category':category,'error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            counts[category]+=1;out.write(json.dumps(row,allow_nan=False)+'\n')
    out={'full_report_sha256':sha(SOURCE/'report.json'),'full_audit_sha256':sha(SOURCE/'audit_v2.json'),
         'previous_audit_sha256':sha(PREVIOUS/'audit_v2.json'),'script_sha256':sha(__file__),
         'records_sha256':sha(HERE/'records.jsonl'),'previous_failures_excluded':len(prior_failed),
         'new_failures_classified':len(candidates),'categories':dict(counts),'training_ready':False,
         'limitations':['Reuses project FK/decoder and technical tolerances; not independent physics or author filter.',
                        'Unexplained audit failures remain quarantined; no automatic correction or acceptance.']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
