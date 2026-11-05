"""Prepare isolated full-rate soft-anchor solver with provenance checks."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'accad_013267_pyroki_fullrate_20261031'
OLD=ROOT/'pyroki_s3_anchor_tail_20261030'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old=json.loads((OLD/'manifest.json').read_text());helper=OLD/'retarget_helpers/_utils.py'
    check=json.loads((BASE/'audit_report.json').read_text())
    if sha(helper)!=old['anchor_helper_sha256'] or sha(BASE/'report.json')!=check['report_sha256']:
        raise ValueError('prior source changed')
    d=HERE/'retarget_helpers';d.mkdir()
    for n in ('__init__.py','_utils.py'):
        with (d/n).open('xb') as f:f.write((OLD/'retarget_helpers'/n).read_bytes())
    launcher=(BASE/'run.sh').read_text()
    a='accad_013267_pyroki_fullrate_20261031';b='accad_013267_anchor_fullrate_20261031'
    if launcher.count(a)!=1:raise ValueError('unexpected launcher')
    with (HERE/'run.sh').open('x',encoding='utf8',newline='\n') as f:f.write(launcher.replace(a,b))
    with (HERE/'audit.py').open('xb') as f:f.write((BASE/'audit.py').read_bytes())
    info={'helper_sha256':sha(d/'_utils.py'),'helper_source_sha256':sha(helper),
          'baseline_report_sha256':sha(BASE/'report.json'),'baseline_audit_sha256':sha(BASE/'audit_report.json'),
          'driver_sha256':sha(HERE/'run.py'),'audit_sha256':sha(HERE/'audit.py'),
          'anchor_weight':10.,'training_ready':False,
          'limitations':['Single candidate, exploratory soft anchoring and same 120Hz assumptions.',
                         'No physical, caption, source or training acceptance.']}
    with (HERE/'manifest.json').open('x',encoding='utf8') as f:json.dump(info,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
