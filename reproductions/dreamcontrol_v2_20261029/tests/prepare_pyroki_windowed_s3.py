"""Create an isolated S3 window experiment from the audited S2 driver."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def change(s,old,new):
    if s.count(old)!=1:raise ValueError('expected one occurrence: '+old)
    return s.replace(old,new)
def save(path,s):
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(s)

def main():
    baseline=ROOT/'pyroki_grab_windowed_20261030';receipt=json.loads((baseline/'report.json').read_text())
    driver=baseline/'pyroki_grab_windowed_pilot.py';audit=baseline/'audit_pyroki_windowed_pilot.py'
    if sha(driver)!=receipt['script_sha256']:raise ValueError('baseline driver mismatch')
    new_source="ROOT/'grab_s3_human_source_20261030/human_source.npz'"
    old_source="ROOT/'grab_human_robot_pilot_20261030/pilot_arrays.npz'"
    code=change(driver.read_text(),old_source,new_source)
    code=change(code,"if sha(source)!=source_receipt['artifact_sha256'] or sha(source)!=variant_receipt['source_sha256']:",
                "if sha(source)!=source_receipt['artifact_sha256']:")
    save(HERE/'pyroki_grab_windowed_pilot.py',code)
    code=change(audit.read_text(),"run=ROOT/'pyroki_grab_windowed_20261030';report=json.loads((run/'report.json').read_text())",
                "run=Path(__file__).resolve().parent;report=json.loads((run/'report.json').read_text())")
    code=change(code,old_source,new_source)
    code=change(code,"if sorted(counts)!=list(range(542,660)):raise ValueError('coverage mismatch')",
                "if sorted(counts)!=list(range(int(source_frames[2]),int(source_frames[-1])+1)):raise ValueError('coverage mismatch')")
    save(HERE/'audit_pyroki_windowed_pilot.py',code)
    src=ROOT/'grab_s3_human_source_20261030/report.json'
    info={'source_receipt_sha256':sha(src),'s2_driver_sha256':sha(driver),'s2_audit_sha256':sha(audit),
          's3_driver_sha256':sha(HERE/'pyroki_grab_windowed_pilot.py'),
          's3_audit_sha256':sha(HERE/'audit_pyroki_windowed_pilot.py'),
          'description':'S3 apple sequence with same solver and window configuration; only source path and general source-dependent coverage checks changed.'}
    save(HERE/'manifest.json',json.dumps(info,indent=2));print(json.dumps(info),flush=True)

if __name__=='__main__':main()
