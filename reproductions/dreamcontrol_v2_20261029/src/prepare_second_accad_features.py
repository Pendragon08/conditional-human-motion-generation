"""Adapt the audited 013267 feature pipeline to the 004347 candidate."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SRC=ROOT/'accad_013267_fullrate_candidate_20261031'
DST=ROOT/'accad_004347_fullrate_candidate_20261031'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old=json.loads((SRC/'report.json').read_text());audit=json.loads((SRC/'audit_report.json').read_text())
    if sha(SRC/'report.json')!=audit['candidate_report_sha256'] or sha(SRC/'export.py')!=old['script_sha256']:
        raise ValueError('prior feature source changed')
    DST.mkdir();changes=[]
    for name in ('export.py','audit.py','run_accad_candidate_after_windows.sh'):
        text=(SRC/name).read_text();count=text.count('013267')
        if count<1:raise ValueError('missing motion ID in '+name)
        with (DST/name).open('x',encoding='utf8',newline='\n') as f:f.write(text.replace('013267','004347'))
        changes.append({'file':name,'old_sha256':sha(SRC/name),'new_sha256':sha(DST/name),'replacements':count})
    for name in ('blend_robot_windows.py','resample_g1_27.py','feature263.py','g1_fk.py'):
        with (DST/name).open('xb') as f:f.write((SRC/name).read_bytes())
    info={'previous_report_sha256':sha(SRC/'report.json'),'previous_audit_sha256':sha(SRC/'audit_report.json'),
          'changes':changes,'training_ready':False,
          'limitations':['Candidate inputs and physical quality must be checked independently.',
                         'No change to earlier data or checkpoint.']}
    with (DST/'prepare_manifest.json').open('x',encoding='utf8') as f:json.dump(info,f,indent=2);f.write('\n')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
