"""Continue an already-audited anchored run in a separate immutable directory."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def change(s,a,b):
    if s.count(a)!=1:raise ValueError('expected one occurrence: '+a)
    return s.replace(a,b)
def save(p,s):
    with p.open('x',encoding='utf8',newline='\n') as f:f.write(s)

def main():
    base=ROOT/'pyroki_s3_overlap_anchor_20261030'
    report=json.loads((base/'report.json').read_text());audit=json.loads((base/'audit_report.json').read_text())
    driver=base/'pyroki_overlap_anchor_pilot.py';helper=base/'retarget_helpers/_utils.py';rows=base/'windows.jsonl'
    if sha(driver)!=report['script_sha256'] or sha(helper)!=report['helper_sha256'] or sha(rows)!=report['rows_sha256'] or sha(base/'report.json')!=audit['report_sha256']:
        raise ValueError('prior code, output or audit mismatch')
    last=json.loads(rows.read_text().splitlines()[-1]);seed=base/'window_05.npz'
    if last['window']!=5 or sha(seed)!=last['artifact_sha256']:raise ValueError('seed mismatch')
    d=HERE/'retarget_helpers';d.mkdir(exist_ok=False);save(d/'__init__.py','');save(d/'_utils.py',helper.read_text())
    code=driver.read_text()
    code=change(code,"first=base/rows[0]['artifact']\n    if sha(first)!=rows[0]['artifact_sha256']:raise ValueError('seed mismatch')",
                "first=ROOT/'pyroki_s3_overlap_anchor_20261030/window_05.npz'\n    if sha(first)!='"+last['artifact_sha256']+"':raise ValueError('seed mismatch')")
    code=change(code,'for i in range(1,6):','for i in range(6,29):')
    code=change(code,"if len(common)!=3 or not np.array_equal(new,np.arange(3)):raise ValueError('unexpected overlap')",
                "if len(common) not in (3,4) or not np.array_equal(new,np.arange(len(common))):raise ValueError('unexpected overlap')")
    code=change(code,'Five windows of one S3 action, initialized by the previously computed first window',
                '23 windows of one S3 action, continued from the previously audited anchored fifth window')
    save(HERE/'pyroki_overlap_anchor_pilot.py',code)
    doc={'baseline_helper_sha256':sha(helper),'anchor_helper_sha256':sha(d/'_utils.py'),
         'prior_report_sha256':sha(base/'report.json'),'prior_audit_sha256':sha(base/'audit_report.json'),
         'seed_sha256':sha(seed),'driver_sha256':sha(HERE/'pyroki_overlap_anchor_pilot.py'),
         'description':'Independent continuation from audited window 5 through window 28; last overlap has four frames. Same experimental anchor weight and solver.'}
    save(HERE/'manifest.json',json.dumps(doc,indent=2));print(json.dumps(doc),flush=True)

if __name__=='__main__':main()
