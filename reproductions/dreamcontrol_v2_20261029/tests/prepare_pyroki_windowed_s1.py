"""Create isolated S1 pour pilot from the audited S3 window implementation."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    old=ROOT/'pyroki_grab_s3_windowed_20261030';manifest=json.loads((old/'manifest.json').read_text());report=json.loads((old/'report.json').read_text())
    driver=old/'pyroki_grab_windowed_pilot.py';audit=old/'audit_pyroki_windowed_pilot.py'
    if sha(driver)!=manifest['s3_driver_sha256'] or sha(driver)!=report['script_sha256'] or sha(audit)!=manifest['s3_audit_sha256']:raise ValueError('previous code changed')
    old_path="ROOT/'grab_s3_human_source_20261030/human_source.npz'";new_path="ROOT/'grab_s1_pour_human_source_20261030/human_source.npz'"
    for path in (driver,audit):
        text=path.read_text()
        if text.count(old_path)!=1:raise ValueError('unexpected source placeholder')
        with (HERE/path.name).open('x',encoding='utf8',newline='\n') as f:f.write(text.replace(old_path,new_path))
    source=ROOT/'grab_s1_pour_human_source_20261030/report.json'
    source_report=json.loads(source.read_text())
    if sha(source.parent/'human_source.npz')!=source_report['artifact_sha256']:raise ValueError('source changed')
    out={'s3_driver_sha256':sha(driver),'s3_audit_sha256':sha(audit),'s1_source_receipt_sha256':sha(source),
         's1_driver_sha256':sha(HERE/driver.name),'s1_audit_sha256':sha(HERE/audit.name),
         'description':'Only the selected source path changes; same v1 PyRoki approximate window settings and audit. No automatic training acceptance.'}
    with (HERE/'manifest.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
